from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
import os
from pathlib import Path
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FINAL_EMBEDDINGS_ROOT = ROOT / "data" / "processed" / "model_runs" / "final_embeddings_train_validation"
DEFAULT_MODEL_ID = ""
TAG_CACHE_PATH = Path(__file__).resolve().parent / "data" / "tags.json"
CATALOG_CACHE_PATH = Path(__file__).resolve().parent / "data" / "catalog.json"
QUERY_MODES = ("demo", "evaluation")


def normalize_search_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.casefold().split())


@dataclass(frozen=True)
class Track:
    spotify_id: str
    name: str
    artist: str
    tags: tuple[str, ...] = ()
    split: str = "unknown"

    @property
    def spotify_url(self) -> str:
        return f"https://open.spotify.com/track/{self.spotify_id}"

    @property
    def spotify_embed_url(self) -> str:
        return f"https://open.spotify.com/embed/track/{self.spotify_id}?utm_source=generator"

    def as_dict(self) -> dict[str, str]:
        return {
            "spotify_id": str(self.spotify_id),
            "name": str(self.name),
            "artist": str(self.artist),
            "tags": list(self.tags),
            "split": str(self.split),
            "spotify_url": self.spotify_url,
            "spotify_embed_url": self.spotify_embed_url,
        }


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    path: Path
    embedding_key: str
    description: str
    available: bool = True
    missing_reason: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "description": self.description,
            "available": self.available,
            "missing_reason": self.missing_reason,
        }


KNOWN_MODELS: tuple[ModelSpec, ...] = ()


def _slugify_model_id(value: str) -> str:
    slug = []
    previous_dash = False
    for char in value.casefold():
        if char.isalnum():
            slug.append(char)
            previous_dash = False
            continue
        if previous_dash:
            continue
        slug.append("_")
        previous_dash = True
    return "".join(slug).strip("_") or "model"


def _embedding_key_from_blob(blob) -> str | None:
    for key in ("embeddings", "song_embeddings", "mix_embeddings", "stem_embeddings"):
        if key in blob.files:
            return key
    return None


def _discover_local_models() -> list[ModelSpec]:
    known_paths = {spec.path.resolve() for spec in KNOWN_MODELS}
    discovered: list[ModelSpec] = []
    data_root = FINAL_EMBEDDINGS_ROOT
    if not data_root.exists():
        return discovered

    for path in sorted(data_root.rglob("*.npz")):
        resolved = path.resolve()
        if resolved in known_paths:
            continue
        try:
            blob = np.load(path, allow_pickle=True)
        except Exception:
            continue
        embedding_key = _embedding_key_from_blob(blob)
        if embedding_key is None or "spotify_id" not in blob.files:
            continue
        relative_parent = path.parent.relative_to(data_root)
        relative_label = str(relative_parent) if relative_parent != Path(".") else path.stem
        model_id = _slugify_model_id(str(relative_parent))
        discovered.append(
            ModelSpec(
                model_id=model_id,
                label=relative_label.replace("_", " ").replace("/", " / "),
                path=path,
                embedding_key=embedding_key,
                description="Local embedding model ready for comparison.",
            )
        )
    return discovered


class RecommenderIndex:
    def __init__(
        self,
        spec: ModelSpec,
        metadata_lookup: dict[str, tuple[str, str]] | None = None,
        tags_lookup: dict[str, tuple[str, ...]] | None = None,
        split_lookup: dict[str, str] | None = None,
    ) -> None:
        self.spec = spec
        if not spec.path.exists():
            raise FileNotFoundError(f"Expected embeddings at {spec.path}")

        blob = np.load(spec.path, allow_pickle=True)
        spotify_ids = blob["spotify_id"].astype(str)
        self.embedding_spaces = self._load_embedding_spaces(blob, spec.embedding_key)
        self.default_space = self._default_space()

        metadata_lookup = metadata_lookup or {}
        tags_lookup = tags_lookup or {}
        split_lookup = split_lookup or {}
        names = blob["name"].astype(str) if "name" in blob.files else None
        artists = blob["artist"].astype(str) if "artist" in blob.files else None

        self.tracks = []
        for idx, spotify_id in enumerate(spotify_ids):
            if names is not None and artists is not None:
                name = str(names[idx])
                artist = str(artists[idx])
            else:
                name, artist = metadata_lookup.get(str(spotify_id), ("Unknown track", "Metadata unavailable"))
            tags = tags_lookup.get(str(spotify_id), ())
            split = split_lookup.get(str(spotify_id), "unknown")
            self.tracks.append(
                Track(spotify_id=str(spotify_id), name=name, artist=artist, tags=tuple(tags), split=split)
            )

        self.lookup = {track.spotify_id: idx for idx, track in enumerate(self.tracks)}
        self.query_indices = {
            "demo": np.arange(len(self.tracks), dtype=np.int32),
            "evaluation": np.asarray(
                [idx for idx, track in enumerate(self.tracks) if track.split == "test"],
                dtype=np.int32,
            ),
        }
        self.artist_lookup: dict[str, list[Track]] = {}
        for track in self.tracks:
            artist_key = track.artist.strip().casefold()
            if not artist_key or artist_key == "metadata unavailable":
                continue
            self.artist_lookup.setdefault(artist_key, []).append(track)
        self.search_text = np.asarray(
            [
                normalize_search_text(
                    f"{track.name} {track.artist} {track.spotify_id} {' '.join(track.tags)}"
                )
                for track in self.tracks
            ],
            dtype=object,
        )
        self.normalized_tags = np.asarray(
            [tuple(normalize_search_text(tag) for tag in track.tags) for track in self.tracks],
            dtype=object,
        )
        self.tag_catalog = self._build_tag_catalog()

    def _shared_tags(self, left: Track, right: Track, limit: int = 4) -> list[str]:
        right_tags = set(right.tags)
        if not right_tags:
            return []
        return [tag for tag in left.tags if tag in right_tags][:limit]

    def _unique_tags(self, source: Track, other: Track, limit: int = 3) -> list[str]:
        other_tags = set(other.tags)
        return [tag for tag in source.tags if tag not in other_tags][:limit]

    def _recommendation_reasons(self, query_track: Track, candidate_track: Track, similarity: float) -> dict[str, object]:
        shared_tags = self._shared_tags(query_track, candidate_track)
        query_only_tags = self._unique_tags(query_track, candidate_track)
        candidate_only_tags = self._unique_tags(candidate_track, query_track)
        same_artist = (
            bool(query_track.artist.strip())
            and query_track.artist.strip().casefold() == candidate_track.artist.strip().casefold()
        )
        strength_label = "High"
        strength_short = "Strong audio match"
        if similarity < 0.88:
            strength_label = "Medium"
            strength_short = "Solid audio match"
        if similarity < 0.72:
            strength_label = "Light"
            strength_short = "Looser audio match"
        overlap_label = f"{len(shared_tags)} shared tag{'s' if len(shared_tags) != 1 else ''}" if shared_tags else "No shared tags"

        primary_reason = f"{strength_short}, {overlap_label.lower()}"
        if same_artist:
            primary_reason = f"{primary_reason}, same artist"

        return {
            "shared_tags": shared_tags,
            "query_only_tags": query_only_tags,
            "candidate_only_tags": candidate_only_tags,
            "shared_tag_count": len(shared_tags),
            "same_artist": same_artist,
            "similarity": round(float(similarity), 4),
            "similarity_percent": int(max(0, min(100, round(float(similarity) * 100)))),
            "strength_label": strength_label,
            "strength_short": strength_short,
            "overlap_label": overlap_label,
            "artist_label": "Same artist" if same_artist else "Different artist",
            "primary_reason": primary_reason,
        }

    def _normalize_mode(self, mode: str | None) -> str:
        if not mode:
            return "demo"
        normalized = mode.strip().casefold()
        if normalized not in QUERY_MODES:
            raise ValueError(normalized)
        return normalized

    def queryable_count(self, mode: str | None = None) -> int:
        normalized = self._normalize_mode(mode)
        return int(len(self.query_indices[normalized]))

    def _query_indices_for_mode(self, mode: str | None = None) -> np.ndarray:
        return self.query_indices[self._normalize_mode(mode)]

    def _validate_query_track(self, spotify_id: str, mode: str | None = None) -> int:
        idx = self.lookup.get(spotify_id)
        if idx is None:
            raise KeyError(spotify_id)

        normalized = self._normalize_mode(mode)
        if normalized == "evaluation" and self.tracks[idx].split != "test":
            raise PermissionError(spotify_id)
        return idx

    def _load_embedding_spaces(self, blob, preferred_key: str) -> dict[str, np.ndarray]:
        key_map = {
            "embeddings": "song",
            "song_embeddings": "song",
            "mix_embeddings": "mix",
            "stem_embeddings": "stem",
        }
        spaces: dict[str, np.ndarray] = {}
        for key, space_name in key_map.items():
            if key not in blob.files:
                continue
            values = self._normalize_embeddings(np.asarray(blob[key], dtype=np.float32))
            spaces[space_name] = values

        if not spaces:
            raise ValueError(f"No supported embedding arrays found in {self.spec.path}")

        preferred_space = key_map.get(preferred_key)
        if preferred_space and preferred_space in spaces:
            spaces = {preferred_space: spaces[preferred_space], **{k: v for k, v in spaces.items() if k != preferred_space}}
        return spaces

    def _default_space(self) -> str:
        if "song" in self.embedding_spaces:
            return "song"
        return next(iter(self.embedding_spaces))

    def available_spaces(self) -> list[str]:
        ordered = [space for space in ("song", "mix", "stem") if space in self.embedding_spaces]
        if "mix" in self.embedding_spaces and "stem" in self.embedding_spaces:
            ordered.append("blend")
        return ordered

    def _normalize_embeddings(self, values: np.ndarray) -> np.ndarray:
        sanitized = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(sanitized, axis=1, keepdims=True).clip(min=1e-8)
        normalized = sanitized / norms
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _embeddings_for_space(self, space: str, blend: float = 0.5) -> np.ndarray:
        if space == "blend":
            if "mix" not in self.embedding_spaces or "stem" not in self.embedding_spaces:
                raise ValueError(space)
            alpha = float(min(max(blend, 0.0), 1.0))
            values = (1.0 - alpha) * self.embedding_spaces["mix"] + alpha * self.embedding_spaces["stem"]
            return self._normalize_embeddings(values)

        if space not in self.embedding_spaces:
            raise ValueError(space)
        return self.embedding_spaces[space]

    def _project_points(
        self,
        vectors: np.ndarray,
        n_components: int,
        seed: int,
    ) -> np.ndarray:
        del seed

        if len(vectors) <= 1:
            return np.zeros((len(vectors), n_components), dtype=np.float32)

        if len(vectors) == 2:
            coords = np.zeros((2, n_components), dtype=np.float32)
            coords[0, 0] = -1.0
            coords[1, 0] = 1.0
            return coords

        # Classical MDS on cosine distance preserves pairwise geometry more faithfully
        # than projecting the raw embedding vectors with a linear SVD.
        similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
        distances = 1.0 - similarity
        squared = distances ** 2
        count = len(vectors)
        identity = np.eye(count, dtype=np.float32)
        centering = identity - np.full((count, count), 1.0 / count, dtype=np.float32)
        gram = -0.5 * centering @ squared @ centering

        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        positive = np.clip(eigvals[:n_components], a_min=0.0, a_max=None).astype(np.float32)
        coords = eigvecs[:, :n_components] * np.sqrt(positive)
        return np.asarray(coords, dtype=np.float32)

    def search(self, query: str, limit: int, mode: str | None = None) -> list[dict[str, str]]:
        eligible_idx = self._query_indices_for_mode(mode)
        if eligible_idx.size == 0:
            return []

        query = normalize_search_text(query)
        if not query:
            return self.random(limit, mode=mode)

        exact_tag_idx: list[int] = []
        partial_tag_idx: list[int] = []
        text_idx: list[int] = []

        for idx in eligible_idx:
            track_idx = int(idx)
            tags = self.normalized_tags[track_idx]
            if any(tag == query for tag in tags):
                exact_tag_idx.append(track_idx)
                continue
            if any(query in tag for tag in tags):
                partial_tag_idx.append(track_idx)
                continue
            if query in str(self.search_text[track_idx]):
                text_idx.append(track_idx)

        ordered_idx = np.asarray(exact_tag_idx + partial_tag_idx + text_idx, dtype=np.int32)[:limit]
        idx = ordered_idx
        return [self.tracks[int(i)].as_dict() for i in idx]

    def _build_tag_catalog(self) -> dict[str, list[dict[str, object]]]:
        catalogs: dict[str, list[dict[str, object]]] = {}
        for mode_name in QUERY_MODES:
            counts: Counter[str] = Counter()
            display_lookup: dict[str, str] = {}
            for idx in self.query_indices[mode_name]:
                track = self.tracks[int(idx)]
                for raw_tag, normalized_tag in zip(track.tags, self.normalized_tags[int(idx)], strict=False):
                    if not normalized_tag:
                        continue
                    counts[normalized_tag] += 1
                    display_lookup.setdefault(normalized_tag, raw_tag)
            catalogs[mode_name] = [
                {
                    "tag": display_lookup[tag],
                    "normalized_tag": tag,
                    "count": int(count),
                }
                for tag, count in sorted(counts.items(), key=lambda item: (-item[1], display_lookup[item[0]]))
            ]
        return catalogs

    def search_tags(self, query: str, limit: int = 12, mode: str | None = None) -> list[dict[str, object]]:
        normalized_mode = self._normalize_mode(mode)
        normalized_query = normalize_search_text(query)
        tags = self.tag_catalog[normalized_mode]
        if not normalized_query:
            return tags[:limit]

        exact = [row for row in tags if row["normalized_tag"] == normalized_query]
        partial = [row for row in tags if normalized_query in str(row["normalized_tag"]) and row not in exact]
        return (exact + partial)[:limit]

    def random_tags(self, limit: int = 1, mode: str | None = None) -> list[dict[str, object]]:
        normalized_mode = self._normalize_mode(mode)
        tags = self.tag_catalog[normalized_mode]
        if not tags:
            return []

        preferred = [row for row in tags if int(row.get("count", 0)) >= 25]
        pool = preferred or tags
        count = min(max(limit, 1), len(tags))
        idx = np.random.default_rng().choice(len(pool), size=min(count, len(pool)), replace=False)
        return [pool[int(i)] for i in np.atleast_1d(idx)]

    def tracks_for_tag(self, tag: str, limit: int = 12, mode: str | None = None) -> list[dict[str, str]]:
        normalized_mode = self._normalize_mode(mode)
        normalized_tag = normalize_search_text(tag)
        if not normalized_tag:
            return self.random(limit, mode=normalized_mode)

        eligible_idx = self._query_indices_for_mode(normalized_mode)
        matched_idx = [
            int(idx)
            for idx in eligible_idx
            if any(track_tag == normalized_tag for track_tag in self.normalized_tags[int(idx)])
        ]
        if not matched_idx:
            return []
        selected_idx = np.asarray(matched_idx[:limit], dtype=np.int32)
        return [self.tracks[int(i)].as_dict() for i in selected_idx]

    def random(self, limit: int, mode: str | None = None) -> list[dict[str, str]]:
        eligible_idx = self._query_indices_for_mode(mode)
        if eligible_idx.size == 0:
            return []

        count = min(max(limit, 1), len(eligible_idx))
        idx = np.random.default_rng().choice(eligible_idx, size=count, replace=False)
        return [self.tracks[int(i)].as_dict() for i in idx]

    def artist_profile(self, artist: str, exclude_spotify_id: str | None = None, limit: int = 12) -> dict[str, object]:
        artist_key = artist.strip().casefold()
        tracks = self.artist_lookup.get(artist_key, [])
        filtered = [track for track in tracks if track.spotify_id != exclude_spotify_id]
        return {
            "artist": artist,
            "count": len(filtered),
            "tracks": [track.as_dict() for track in filtered[:limit]],
        }

    def distance(
        self,
        spotify_id_a: str,
        spotify_id_b: str,
        space: str | None = None,
        blend: float = 0.5,
    ) -> dict[str, object]:
        idx_a = self.lookup.get(spotify_id_a)
        idx_b = self.lookup.get(spotify_id_b)
        if idx_a is None:
            raise KeyError(spotify_id_a)
        if idx_b is None:
            raise KeyError(spotify_id_b)

        chosen_space = space or self.default_space
        embeddings = self._embeddings_for_space(chosen_space, blend=blend)
        similarity = float(np.dot(embeddings[idx_a], embeddings[idx_b]))

        return {
            "model": {**self.spec.as_dict(), "spaces": self.available_spaces()},
            "space": chosen_space,
            "blend": round(float(blend), 2) if chosen_space == "blend" else None,
            "track_a": self.tracks[idx_a].as_dict(),
            "track_b": self.tracks[idx_b].as_dict(),
            "cosine_similarity": round(similarity, 4),
            "cosine_distance": round(1.0 - similarity, 4),
        }

    def neighborhood_map(
        self,
        spotify_id: str,
        space: str | None = None,
        blend: float = 0.5,
        limit: int = 20,
    ) -> dict[str, object]:
        idx = self.lookup.get(spotify_id)
        if idx is None:
            raise KeyError(spotify_id)

        chosen_space = space or self.default_space
        embeddings = self._embeddings_for_space(chosen_space, blend=blend)
        scores = embeddings @ embeddings[idx]
        scores[idx] = -np.inf

        limit = min(max(limit, 5), len(self.tracks) - 1)
        neighbor_idx = np.argpartition(-scores, kth=limit - 1)[:limit]
        selected_idx = np.concatenate(([idx], neighbor_idx))
        vectors = embeddings[selected_idx]
        coords = self._project_points(vectors, n_components=2, seed=7)

        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)

        xs = coords[:, 0]
        ys = coords[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        def scale(value: float, lower: float, upper: float) -> float:
            if abs(upper - lower) < 1e-8:
                return 0.5
            return float((value - lower) / (upper - lower))

        points = []
        for point_idx, track_idx in enumerate(selected_idx):
            track = self.tracks[int(track_idx)].as_dict()
            points.append(
                {
                    **track,
                    "x": round(scale(xs[point_idx], x_min, x_max), 4),
                    "y": round(scale(ys[point_idx], y_min, y_max), 4),
                    "is_query": point_idx == 0,
                    "similarity": None if point_idx == 0 else round(float(scores[int(track_idx)]), 4),
                }
            )

        return {
            "model": {**self.spec.as_dict(), "spaces": self.available_spaces()},
            "space": chosen_space,
            "blend": round(float(blend), 2) if chosen_space == "blend" else None,
            "points": points,
        }

    def manifold_projection(
        self,
        spotify_id: str | None = None,
        space: str | None = None,
        blend: float = 0.5,
        sample_limit: int = 220,
        neighbor_limit: int = 24,
        seed: int = 7,
    ) -> dict[str, object]:
        chosen_space = space or self.default_space
        embeddings = self._embeddings_for_space(chosen_space, blend=blend)

        track_idx: int | None = None
        if spotify_id is not None:
            track_idx = self.lookup.get(spotify_id)
            if track_idx is None:
                raise KeyError(spotify_id)

        sample_limit = min(max(sample_limit, 1), len(self.tracks))
        rng = np.random.default_rng(seed)

        selected_set: set[int] = set()
        if track_idx is not None:
            scores = embeddings @ embeddings[track_idx]
            scores[track_idx] = -np.inf
            selected_set.add(track_idx)
            if sample_limit > 1 and len(self.tracks) > 1:
                neighbor_limit = min(max(neighbor_limit, 1), max(sample_limit - 1, 1), len(self.tracks) - 1)
                neighbor_idx = np.argpartition(-scores, kth=neighbor_limit - 1)[:neighbor_limit]
                selected_set.update(int(idx) for idx in neighbor_idx)

        remaining = sample_limit - len(selected_set)
        if remaining > 0:
            candidates = np.array([idx for idx in range(len(self.tracks)) if idx not in selected_set], dtype=np.int32)
            if len(candidates) <= remaining:
                selected_set.update(int(idx) for idx in candidates)
            else:
                sampled = rng.choice(candidates, size=remaining, replace=False)
                selected_set.update(int(idx) for idx in sampled)

        selected_idx = np.array(sorted(selected_set), dtype=np.int32)
        vectors = embeddings[selected_idx]
        coords = self._project_points(vectors, n_components=3, seed=seed)
        if coords.shape[1] < 3:
            zeros = np.zeros((coords.shape[0], 3 - coords.shape[1]), dtype=coords.dtype)
            coords = np.concatenate([coords, zeros], axis=1)

        max_abs = np.abs(coords).max(axis=0)
        max_abs[max_abs < 1e-8] = 1.0
        coords = coords / max_abs

        neighbor_set: set[int] = set()
        if track_idx is not None and len(selected_idx) > 1:
            local_scores = embeddings[selected_idx] @ embeddings[track_idx]
            local_rank = np.argsort(-local_scores)
            for order in local_rank[1 : min(neighbor_limit + 1, len(local_rank))]:
                neighbor_set.add(int(selected_idx[int(order)]))

        points = []
        for pos, idx in enumerate(selected_idx):
            track = self.tracks[int(idx)].as_dict()
            role = "context"
            if track_idx is not None and int(idx) == track_idx:
                role = "query"
            elif int(idx) in neighbor_set:
                role = "neighbor"
            points.append(
                {
                    **track,
                    "x": round(float(coords[pos, 0]), 5),
                    "y": round(float(coords[pos, 1]), 5),
                    "z": round(float(coords[pos, 2]), 5),
                    "role": role,
                }
            )

        return {
            "model": {**self.spec.as_dict(), "spaces": self.available_spaces()},
            "space": chosen_space,
            "blend": round(float(blend), 2) if chosen_space == "blend" else None,
            "query": self.tracks[track_idx].as_dict() if track_idx is not None else None,
            "points": points,
            "counts": {
                "total": len(points),
                "neighbors": len(neighbor_set),
                "context": len(points) - len(neighbor_set) - (1 if track_idx is not None else 0),
            },
        }

    def global_stats(
        self,
        space: str | None = None,
        blend: float = 0.5,
        pair_limit: int = 10,
        song_limit: int = 10,
        neighborhood_k: int = 8,
        chunk_size: int = 512,
    ) -> dict[str, object]:
        chosen_space = space or self.default_space
        embeddings = self._embeddings_for_space(chosen_space, blend=blend)
        n_tracks = len(self.tracks)
        pair_limit = min(max(pair_limit, 3), 25)
        song_limit = min(max(song_limit, 3), 25)
        neighborhood_k = min(max(neighborhood_k, 3), max(n_tracks - 1, 1))

        centroid = embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True).clip(min=1e-8)
        centroid_scores = (embeddings @ centroid[0]).astype(np.float32)

        top_pairs: list[tuple[float, int, int]] = []
        neighbor_strength = np.empty(n_tracks, dtype=np.float32)
        for start in range(0, n_tracks, chunk_size):
            end = min(start + chunk_size, n_tracks)
            scores = embeddings[start:end] @ embeddings.T
            row_idx = np.arange(start, end)
            scores[np.arange(end - start), row_idx] = -np.inf

            local_k = min(pair_limit, n_tracks - 1)
            top_idx = np.argpartition(-scores, kth=local_k - 1, axis=1)[:, :local_k]
            top_scores = np.take_along_axis(scores, top_idx, axis=1)
            for offset in range(end - start):
                source_idx = int(row_idx[offset])
                order = np.argsort(-top_scores[offset])
                for candidate_pos in order:
                    target_idx = int(top_idx[offset, candidate_pos])
                    if source_idx < target_idx:
                        top_pairs.append((float(top_scores[offset, candidate_pos]), source_idx, target_idx))

            local_neighbor_k = min(neighborhood_k, n_tracks - 1)
            strongest_idx = np.argpartition(-scores, kth=local_neighbor_k - 1, axis=1)[:, :local_neighbor_k]
            strongest_scores = np.take_along_axis(scores, strongest_idx, axis=1)
            neighbor_strength[start:end] = strongest_scores.mean(axis=1)

        top_pairs.sort(key=lambda row: row[0], reverse=True)
        closest_pairs = []
        seen_pairs: set[tuple[int, int]] = set()
        for similarity, idx_a, idx_b in top_pairs:
            key = (idx_a, idx_b)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            closest_pairs.append(
                {
                    "similarity": round(similarity, 4),
                    "distance": round(1.0 - similarity, 4),
                    "track_a": self.tracks[idx_a].as_dict(),
                    "track_b": self.tracks[idx_b].as_dict(),
                }
            )
            if len(closest_pairs) >= pair_limit:
                break

        central_idx = np.argsort(-neighbor_strength)[:song_limit]
        outlier_idx = np.argsort(centroid_scores)[:song_limit]
        centroid_idx = np.argsort(-centroid_scores)[:song_limit]

        def pack_song(idx: int, score_name: str, score: float) -> dict[str, object]:
            row = self.tracks[int(idx)].as_dict()
            row[score_name] = round(float(score), 4)
            return row

        return {
            "model": {**self.spec.as_dict(), "spaces": self.available_spaces()},
            "space": chosen_space,
            "blend": round(float(blend), 2) if chosen_space == "blend" else None,
            "counts": {
                "tracks": n_tracks,
                "pair_candidates": len(top_pairs),
            },
            "closest_pairs": closest_pairs,
            "central_songs": [pack_song(idx, "neighbor_score", neighbor_strength[int(idx)]) for idx in central_idx],
            "outlier_songs": [pack_song(idx, "centroid_similarity", centroid_scores[int(idx)]) for idx in outlier_idx],
            "centroid_songs": [pack_song(idx, "centroid_similarity", centroid_scores[int(idx)]) for idx in centroid_idx],
        }

    def recommend(
        self,
        spotify_id: str,
        limit: int,
        space: str | None = None,
        blend: float = 0.5,
        mode: str | None = None,
    ) -> dict[str, object]:
        normalized_mode = self._normalize_mode(mode)
        idx = self._validate_query_track(spotify_id, normalized_mode)

        chosen_space = space or self.default_space
        embeddings = self._embeddings_for_space(chosen_space, blend=blend)
        scores = embeddings @ embeddings[idx]
        scores[idx] = -np.inf

        limit = min(max(limit, 1), len(self.tracks) - 1)
        top_idx = np.argpartition(-scores, kth=limit - 1)[:limit]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        recommendations = []
        for rec_idx in top_idx:
            row = self.tracks[int(rec_idx)].as_dict()
            similarity = round(float(scores[int(rec_idx)]), 4)
            row["similarity"] = similarity
            row["why"] = self._recommendation_reasons(self.tracks[idx], self.tracks[int(rec_idx)], similarity)
            recommendations.append(row)

        return {
            "model": {**self.spec.as_dict(), "spaces": self.available_spaces()},
            "mode": normalized_mode,
            "query_pool_size": self.queryable_count(normalized_mode),
            "space": chosen_space,
            "blend": round(float(blend), 2) if chosen_space == "blend" else None,
            "query": self.tracks[idx].as_dict(),
            "recommendations": recommendations,
        }


def _default_override_spec() -> ModelSpec | None:
    override = os.getenv("SONG_RECOMMENDER_EMBEDDINGS_PATH")
    if not override:
        return None
    return ModelSpec(
        model_id="override",
        label="Override Model",
        path=Path(override),
        embedding_key=os.getenv("SONG_RECOMMENDER_EMBEDDING_KEY", "embeddings"),
        description="Loaded from SONG_RECOMMENDER_EMBEDDINGS_PATH.",
    )


def available_models() -> list[ModelSpec]:
    models: list[ModelSpec] = []
    seen: set[str] = set()

    override = _default_override_spec()
    if override is not None and override.path.exists():
        models.append(override)
        seen.add(override.model_id)

    for spec in _discover_local_models():
        if spec.model_id in seen:
            continue
        models.append(spec)
        seen.add(spec.model_id)

    return models


def resolve_model(model_id: str | None) -> ModelSpec:
    models = available_models()
    if not models:
        raise FileNotFoundError("No embedding models found.")

    if not model_id:
        return next((model for model in models if model.model_id == DEFAULT_MODEL_ID), models[0])

    for model in models:
        if model.model_id == model_id:
            if not model.available:
                raise KeyError(model_id)
            return model
    raise KeyError(model_id)


def metadata_lookup() -> dict[str, tuple[str, str]]:
    if CATALOG_CACHE_PATH.exists():
        payload = json.loads(CATALOG_CACHE_PATH.read_text())
        lookup: dict[str, tuple[str, str]] = {}
        for spotify_id, row in payload.items():
            if isinstance(row, dict):
                lookup[str(spotify_id)] = (
                    str(row.get("name", "Unknown track")),
                    str(row.get("artist", "Metadata unavailable")),
                )
        if lookup:
            return lookup

    for model in available_models():
        blob = np.load(model.path, allow_pickle=True)
        if "name" not in blob.files or "artist" not in blob.files:
            continue
        lookup: dict[str, tuple[str, str]] = {}
        for spotify_id, name, artist in zip(
            blob["spotify_id"].astype(str),
            blob["name"].astype(str),
            blob["artist"].astype(str),
            strict=True,
        ):
            lookup[str(spotify_id)] = (str(name), str(artist))
        if lookup:
            return lookup
    return {}


def tags_lookup() -> dict[str, tuple[str, ...]]:
    if CATALOG_CACHE_PATH.exists():
        payload = json.loads(CATALOG_CACHE_PATH.read_text())
        lookup: dict[str, tuple[str, ...]] = {}
        for spotify_id, row in payload.items():
            if isinstance(row, dict):
                tags = row.get("tags", [])
                if isinstance(tags, list):
                    lookup[str(spotify_id)] = tuple(str(tag) for tag in tags if str(tag).strip())
        if lookup:
            return lookup

    if not TAG_CACHE_PATH.exists():
        return {}

    payload = json.loads(TAG_CACHE_PATH.read_text())
    lookup: dict[str, tuple[str, ...]] = {}
    for spotify_id, tags in payload.items():
        if isinstance(tags, list):
            lookup[str(spotify_id)] = tuple(str(tag) for tag in tags if str(tag).strip())
    return lookup


def split_lookup() -> dict[str, str]:
    if CATALOG_CACHE_PATH.exists():
        payload = json.loads(CATALOG_CACHE_PATH.read_text())
        lookup: dict[str, str] = {}
        for spotify_id, row in payload.items():
            if isinstance(row, dict):
                split = str(row.get("split", "")).strip().casefold()
                if split in {"train", "val", "test"}:
                    lookup[str(spotify_id)] = split
        if lookup:
            return lookup

    lookup: dict[str, str] = {}
    for spec in KNOWN_MODELS:
        manifest_path = spec.path.parent / "run_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        split = None
        for split_name, key_names in (
            ("test", ("test_split_path",)),
            ("val", ("val_split_path", "validation_split_path")),
        ):
            for key_name in key_names:
                split_path = str(manifest.get(key_name, "")).strip().casefold()
                if split_path.endswith(f"{split_name}.parquet"):
                    split = split_name
                    break
            if split is not None:
                break
        if split is None:
            continue
        try:
            blob = np.load(spec.path, allow_pickle=True)
        except Exception:
            continue
        for spotify_id in blob["spotify_id"].astype(str):
            lookup[str(spotify_id)] = split
    return lookup
