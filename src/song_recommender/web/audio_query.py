from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from demucs.apply import apply_model
from demucs.pretrained import get_model
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchvision.models import resnet18
import yaml

from song_recommender.web.recommender import ModelSpec, ROOT

AUDIO_CLIP_SECONDS = 10.0
STEM_ORDER = ("bass", "drums", "other", "vocals")
_PREPROCESSING_CONFIG_PATH = ROOT / "configs" / "preprocessing.yaml"
_SEPARATOR_CACHE: dict[str, object] = {}
_ENCODER_CACHE: dict[str, "LateFusionResnetEncoder"] = {}


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32, copy=False)


def _load_preprocessing_config() -> dict[str, object]:
    with _PREPROCESSING_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _clip_or_pad(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.shape[-1] == target_length:
        return values.astype(np.float32, copy=False)
    if values.shape[-1] > target_length:
        return values[..., :target_length].astype(np.float32, copy=False)
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, target_length - values.shape[-1])
    return np.pad(values, pad_width, mode="constant").astype(np.float32, copy=False)


def _resolve_checkpoint_path(spec: ModelSpec) -> Path:
    manifest_path = spec.path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for model {spec.model_id}: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_label = str(payload.get("run_label") or spec.path.parent.name)
    checkpoint_path = ROOT / "data" / "processed" / "model_runs" / run_label / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for model {spec.model_id}: {checkpoint_path}")
    return checkpoint_path


def _load_separator():
    model = _SEPARATOR_CACHE.get("htdemucs")
    if model is None:
        model = get_model("htdemucs")
        model.cpu()
        model.eval()
        _SEPARATOR_CACHE["htdemucs"] = model
    return model


class LateFusionResnetEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        projection_dim: int = 128,
        pretrained: bool = False,
        imagenet_input_norm: bool = True,
        fusion_alpha_init: float = 0.75,
        drum_alpha_init: float = 0.2,
        num_stems: int = 4,
        stem_dropout_prob: float = 0.0,
        harmonic_indices: list[int] | tuple[int, ...] = (0, 2, 3),
        drum_index: int = 1,
    ) -> None:
        del pretrained, stem_dropout_prob
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embedding_dim)
        self.projection_head = nn.Module()
        self.projection_head.net = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self.imagenet_input_norm = bool(imagenet_input_norm)
        self.harmonic_indices = tuple(int(index) for index in harmonic_indices)
        self.drum_index = int(drum_index)
        self.num_stems = int(num_stems)
        self.register_buffer(
            "harmonic_index_tensor",
            torch.tensor(self.harmonic_indices, dtype=torch.long),
            persistent=False,
        )
        self.fusion_alpha_logit = nn.Parameter(torch.tensor(float(np.log(fusion_alpha_init / (1.0 - fusion_alpha_init))), dtype=torch.float32))
        self.drum_alpha_logit = nn.Parameter(torch.tensor(float(np.log(drum_alpha_init / (1.0 - drum_alpha_init))), dtype=torch.float32))
        self.stem_logits = nn.Parameter(torch.zeros(self.num_stems, dtype=torch.float32))
        self.harmonic_logits = nn.Parameter(torch.zeros(len(self.harmonic_indices), dtype=torch.float32))
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"Expected 4D image tensor, received shape {tuple(image.shape)}")
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        if self.imagenet_input_norm:
            image = (image - self.imagenet_mean) / self.imagenet_std
        return image

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self._prepare_image(image))
        return F.normalize(encoded, dim=-1, eps=1e-8)

    def forward(self, mix: torch.Tensor, stems: torch.Tensor) -> dict[str, torch.Tensor]:
        if stems.dim() != 5:
            raise ValueError(f"Expected 5D stems tensor, received shape {tuple(stems.shape)}")
        batch_size, stem_count = stems.shape[:2]
        if stem_count != self.num_stems:
            raise ValueError(f"Expected {self.num_stems} stems, received {stem_count}")

        mix_embedding = self.encode_image(mix)
        stem_embeddings = self.encode_image(stems.view(batch_size * stem_count, *stems.shape[2:])).view(batch_size, stem_count, -1)

        stem_weights = torch.softmax(self.stem_logits, dim=0).unsqueeze(0).expand(batch_size, -1)
        stem_weights = stem_weights / stem_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        harmonic_bank = stem_embeddings.index_select(1, self.harmonic_index_tensor)
        harmonic_weights = torch.softmax(self.harmonic_logits, dim=0).unsqueeze(0).expand(batch_size, -1)
        harmonic_weights = harmonic_weights * stem_weights.index_select(1, self.harmonic_index_tensor)
        harmonic_weights = torch.where(
            harmonic_weights.sum(dim=1, keepdim=True) == 0,
            torch.ones_like(harmonic_weights),
            harmonic_weights,
        )
        harmonic_weights = harmonic_weights / harmonic_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        harmonic_embedding = torch.sum(harmonic_bank * harmonic_weights.unsqueeze(-1), dim=1)

        drum_embedding = stem_embeddings[:, self.drum_index, :] * stem_weights[:, self.drum_index].unsqueeze(-1)

        mix_embedding = F.normalize(mix_embedding, dim=-1, eps=1e-8)
        harmonic_embedding = F.normalize(harmonic_embedding, dim=-1, eps=1e-8)
        drum_embedding = F.normalize(drum_embedding, dim=-1, eps=1e-8)

        fusion_alpha = torch.sigmoid(self.fusion_alpha_logit)
        drum_alpha = torch.sigmoid(self.drum_alpha_logit)
        song_embedding = F.normalize(
            mix_embedding + fusion_alpha * harmonic_embedding + drum_alpha * drum_embedding,
            dim=-1,
            eps=1e-8,
        )

        return {
            "song": song_embedding,
            "mix": mix_embedding,
            # Final embedding exports store the harmonic branch under the `stem_embeddings` name.
            "stem": harmonic_embedding,
        }


def _load_encoder(spec: ModelSpec) -> LateFusionResnetEncoder:
    cached = _ENCODER_CACHE.get(spec.model_id)
    if cached is not None:
        return cached

    checkpoint_path = _resolve_checkpoint_path(spec)
    payload = torch.load(checkpoint_path, map_location="cpu")
    init_kwargs = dict(payload.get("model_init_kwargs", {}))
    model = LateFusionResnetEncoder(**init_kwargs)
    state_dict = {
        key: value
        for key, value in payload["model_state_dict"].items()
        if not key.startswith("projection_head.")
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    _ENCODER_CACHE[spec.model_id] = model
    return model


def _render_spectrogram_image(spec: np.ndarray, config: dict[str, object], image_size: int) -> np.ndarray:
    image_cfg = config["image_output"]
    flipped = np.flipud(np.clip(spec, 0.0, 1.0).astype(np.float32, copy=False))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        plt.imsave(
            temp_path,
            flipped,
            cmap=str(image_cfg["cmap"]),
            vmin=float(image_cfg["vmin"]),
            vmax=float(image_cfg["vmax"]),
        )
        image = Image.open(temp_path).convert("L")
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
        return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32, copy=False)
    finally:
        temp_path.unlink(missing_ok=True)


def _waveform_to_spectrogram(waveform: np.ndarray, config: dict[str, object], ref_value: float | None = None) -> tuple[np.ndarray, float]:
    audio_cfg = config["audio"]
    mel_cfg = config["mel_spectrogram"]
    norm_cfg = config["normalization"]
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(audio_cfg["sample_rate"]),
        n_fft=int(mel_cfg["n_fft"]),
        hop_length=int(mel_cfg["hop_length"]),
        n_mels=int(mel_cfg["n_mels"]),
        f_min=float(mel_cfg["fmin"]),
        f_max=float(mel_cfg["fmax"]),
        power=float(mel_cfg["power"]),
    )
    mel = transform(torch.as_tensor(waveform, dtype=torch.float32)).cpu().numpy()
    if ref_value is None:
        ref_value = float(np.max(mel))
    mel_db = 10.0 * np.log10(np.clip(mel, 1e-10, None) / max(ref_value, 1e-10))
    if norm_cfg["method"] == "minmax":
        mel_norm = (mel_db + 80.0) / 80.0
        if bool(norm_cfg.get("clip", True)):
            mel_norm = np.clip(mel_norm, 0.0, 1.0)
    else:
        mel_norm = mel_db
    return mel_norm.astype(np.float32, copy=False), ref_value


def _load_clip_audio(path: Path, offset_seconds: float, duration_seconds: float, sample_rate: int) -> np.ndarray:
    with sf.SoundFile(str(path)) as handle:
        source_rate = int(handle.samplerate)
        total_frames = int(handle.frames)
        start_frame = max(0, int(round(offset_seconds * source_rate)))
        frame_count = max(1, int(round(duration_seconds * source_rate)))
        if total_frames < frame_count:
            clip_seconds = total_frames / max(source_rate, 1)
            raise ValueError(
                f"Uploaded clip is too short. Need at least {duration_seconds:.0f} seconds, got {clip_seconds:.2f} seconds."
            )
        if start_frame > max(total_frames - frame_count, 0):
            max_start = max((total_frames - frame_count) / max(source_rate, 1), 0.0)
            raise ValueError(
                f"clip_start_sec is out of range for this file. Choose a start between 0.00 and {max_start:.2f} seconds."
            )
        handle.seek(start_frame)
        audio = handle.read(frames=frame_count, dtype="float32", always_2d=True).T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio_tensor = torch.from_numpy(audio)
    if source_rate != sample_rate:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=source_rate, new_freq=sample_rate)
    target_length = int(round(sample_rate * duration_seconds))
    return _clip_or_pad(audio_tensor.cpu().numpy(), target_length)


def _separate_stems(path: Path, offset_seconds: float, duration_seconds: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    separator = _load_separator()
    sample_rate = int(separator.samplerate)
    audio = _load_clip_audio(path, offset_seconds=offset_seconds, duration_seconds=duration_seconds, sample_rate=sample_rate)
    mix_tensor = torch.from_numpy(audio).unsqueeze(0)
    with torch.inference_mode():
        estimates = apply_model(separator, mix_tensor, shifts=0, split=True, overlap=0.25, progress=False, device="cpu")
    source_lookup = {name: int(index) for index, name in enumerate(separator.sources)}
    stems: dict[str, np.ndarray] = {}
    for stem_name in STEM_ORDER:
        source_index = source_lookup[stem_name]
        stems[stem_name] = estimates[0, source_index].mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
    return audio.mean(axis=0).astype(np.float32, copy=False), stems


def _build_model_inputs(path: Path, offset_seconds: float, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    preprocessing = _load_preprocessing_config()
    separator_sample_rate = int(_load_separator().samplerate)
    mix_waveform, stem_waveforms = _separate_stems(path, offset_seconds=offset_seconds, duration_seconds=AUDIO_CLIP_SECONDS)

    audio_sr = int(preprocessing["audio"]["sample_rate"])
    target_length = int(round(audio_sr * AUDIO_CLIP_SECONDS))
    mix_waveform = torchaudio.functional.resample(torch.from_numpy(mix_waveform), orig_freq=separator_sample_rate, new_freq=audio_sr).cpu().numpy()
    mix_waveform = _clip_or_pad(mix_waveform, target_length)

    stem_specs: list[np.ndarray] = []
    mix_spec, reference = _waveform_to_spectrogram(mix_waveform, preprocessing)
    mix_image = _render_spectrogram_image(mix_spec, preprocessing, image_size)

    for stem_name in STEM_ORDER:
        stem_waveform = torchaudio.functional.resample(
            torch.from_numpy(stem_waveforms[stem_name]),
            orig_freq=separator_sample_rate,
            new_freq=audio_sr,
        ).cpu().numpy()
        stem_waveform = _clip_or_pad(stem_waveform, target_length)
        stem_spec, _ = _waveform_to_spectrogram(stem_waveform, preprocessing, ref_value=reference)
        stem_specs.append(_render_spectrogram_image(stem_spec, preprocessing, image_size))

    mix_tensor = torch.from_numpy(mix_image).unsqueeze(0).unsqueeze(0)
    stems_tensor = torch.from_numpy(np.stack(stem_specs, axis=0)).unsqueeze(1).unsqueeze(0)
    return mix_tensor, stems_tensor


@dataclass(frozen=True)
class UploadedClipEmbeddings:
    query: dict[str, object]
    embeddings: dict[str, np.ndarray]


def embed_uploaded_clip(spec: ModelSpec, file_path: Path, clip_start_sec: float, filename: str | None = None) -> UploadedClipEmbeddings:
    encoder = _load_encoder(spec)
    image_size = 224
    mix_tensor, stems_tensor = _build_model_inputs(file_path, offset_seconds=clip_start_sec, image_size=image_size)
    with torch.inference_mode():
        outputs = encoder(mix_tensor, stems_tensor)
    embeddings = {name: _l2_normalize(tensor[0].cpu().numpy()) for name, tensor in outputs.items()}
    return UploadedClipEmbeddings(
        query={
            "spotify_id": "",
            "name": "Uploaded clip",
            "artist": filename or "Local audio file",
            "tags": [],
            "split": "uploaded",
            "spotify_url": "",
            "spotify_embed_url": "",
            "uploaded": True,
            "clip_start_sec": round(float(clip_start_sec), 2),
            "clip_duration_sec": AUDIO_CLIP_SECONDS,
            "source_filename": filename or "",
        },
        embeddings=embeddings,
    )
