from __future__ import annotations

from pathlib import Path
import random
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from song_recommender.web.audio_query import AUDIO_CLIP_SECONDS, embed_uploaded_clip
from song_recommender.web.evaluation_store import create_session, ensure_schema, save_response
from song_recommender.web.recommender import RecommenderIndex, available_models, metadata_lookup, resolve_model, split_lookup, tags_lookup

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "static" / "index.html"
EVALUATION_PATH = ROOT / "static" / "evaluation.html"
MANIFOLD_PATH = ROOT / "static" / "manifold.html"
STATS_PATH = ROOT / "static" / "stats.html"
CATALOG_CACHE_PATH = ROOT / "data" / "catalog.json"

_metadata_lookup: dict[str, tuple[str, str]] = {}
_tags_lookup: dict[str, tuple[str, ...]] = {}
_split_lookup: dict[str, str] = {}
_lookups_mtime_ns: int | None = None
_index_cache: dict[str, RecommenderIndex] = {}


def refresh_shared_lookups() -> None:
    global _metadata_lookup, _tags_lookup, _split_lookup, _lookups_mtime_ns, _index_cache

    current_mtime_ns = CATALOG_CACHE_PATH.stat().st_mtime_ns if CATALOG_CACHE_PATH.exists() else None
    if _lookups_mtime_ns == current_mtime_ns and _metadata_lookup:
        return

    _metadata_lookup = metadata_lookup()
    _tags_lookup = tags_lookup()
    _split_lookup = split_lookup()
    _lookups_mtime_ns = current_mtime_ns
    _index_cache = {}


def get_index(model_id: str | None) -> RecommenderIndex:
    refresh_shared_lookups()
    spec = resolve_model(model_id)
    if spec.model_id not in _index_cache:
        _index_cache[spec.model_id] = RecommenderIndex(
            spec,
            metadata_lookup=_metadata_lookup,
            tags_lookup=_tags_lookup,
            split_lookup=_split_lookup,
        )
    return _index_cache[spec.model_id]


def _default_model_spec() -> dict[str, object] | None:
    try:
        spec = resolve_model(None)
    except (FileNotFoundError, KeyError):
        return None
    return spec.as_dict()


app = FastAPI(title="DL Song Recommender Deployment")
ensure_schema()


class EvaluationResponseBody(BaseModel):
    session_id: str
    model_id: str
    recommendation_spotify_id: str
    rating: int = Field(..., ge=-1, le=2)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/evaluation")
def evaluation_page() -> FileResponse:
    return FileResponse(EVALUATION_PATH)


@app.get("/manifold")
def manifold_page() -> FileResponse:
    return FileResponse(MANIFOLD_PATH)


def _heldout_split_for_index(index: RecommenderIndex) -> str | None:
    counts = {"test": 0, "val": 0}
    for track in index.tracks:
        if track.split in counts:
            counts[track.split] += 1
    if counts["test"] > 0:
        return "test"
    if counts["val"] > 0:
        return "val"
    return None


def _heldout_query_ids(index: RecommenderIndex, split_name: str) -> list[str]:
    return [track.spotify_id for track in index.tracks if track.split == split_name]


@app.post("/api/evaluation/session")
def start_evaluation_session(request: Request) -> dict[str, object]:
    model_specs = [model for model in available_models() if model.available]
    rng = random.Random()
    rng.shuffle(model_specs)
    session_models: list[dict[str, object]] = []
    session_items: list[dict[str, object]] = []

    for label_index, spec in enumerate(model_specs, start=1):
        index = get_index(spec.model_id)
        split_name = _heldout_split_for_index(index)
        if split_name is None:
            continue
        query_ids = _heldout_query_ids(index, split_name)
        if not query_ids:
            continue
        query_id = rng.choice(query_ids)
        payload = index.recommend(query_id, limit=3)
        session_models.append(
            {
                "blind_label": f"Random Model {label_index}",
                "model": payload["model"],
                "query_split": split_name,
                "query": payload["query"],
                "recommendations": payload["recommendations"],
            }
        )
        for rank, recommendation in enumerate(payload["recommendations"], start=1):
            session_items.append(
                {
                    "model_id": payload["model"]["model_id"],
                    "model_label": payload["model"]["label"],
                    "query_split": split_name,
                    "query_spotify_id": payload["query"]["spotify_id"],
                    "query_name": payload["query"]["name"],
                    "query_artist": payload["query"]["artist"],
                    "recommendation_spotify_id": recommendation["spotify_id"],
                    "recommendation_name": recommendation["name"],
                    "recommendation_artist": recommendation["artist"],
                    "recommendation_rank": rank,
                    "similarity": recommendation.get("similarity"),
                }
            )

    if not session_models:
        raise HTTPException(status_code=503, detail="No held-out evaluation tracks are available for the current models.")

    session_id, created_at = create_session(session_items, user_agent=request.headers.get("user-agent"))
    return {
        "session_id": session_id,
        "created_at": created_at,
        "total_models": len(session_models),
        "total_items": len(session_items),
        "models": session_models,
    }


@app.post("/api/evaluation/response")
def save_evaluation_response(body: EvaluationResponseBody) -> dict[str, object]:
    try:
        counts = save_response(
            session_id=body.session_id,
            model_id=body.model_id,
            recommendation_spotify_id=body.recommendation_spotify_id,
            rating=body.rating,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        **counts,
    }


@app.get("/stats")
def stats_page() -> FileResponse:
    return FileResponse(STATS_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "models_discovered": len(available_models()),
        "default_model": _default_model_spec(),
        "catalog_cache_present": CATALOG_CACHE_PATH.exists(),
        "evaluation_store_ready": True,
    }


@app.get("/health/deep")
def deep_health() -> dict[str, object]:
    index = get_index(None)
    return {
        "status": "ok",
        "default_model": index.spec.model_id,
        "default_space": index.default_space,
        "tracks": len(index.tracks),
        "evaluation_query_tracks": index.queryable_count("evaluation"),
        "tag_cache_loaded": bool(_tags_lookup),
    }


@app.get("/api/models")
def models():
    return {
        "models": [
            {
                **model.as_dict(),
                "track_count": len(get_index(model.model_id).tracks) if model.available else 0,
                "spaces": get_index(model.model_id).available_spaces() if model.available else [],
                "evaluation_query_count": get_index(model.model_id).queryable_count("evaluation") if model.available else 0,
            }
            for model in available_models()
        ]
    }


@app.get("/api/search")
def search(
    q: str = Query(default=""),
    model: str | None = Query(default=None),
    mode: str | None = Query(default="demo"),
    limit: int = Query(default=12, ge=1, le=30),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc
    try:
        results = index.search(q, limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {exc.args[0]}") from exc
    return {"model": index.spec.as_dict(), "mode": mode or "demo", "results": results}


@app.get("/api/tags/search")
def search_tags(
    q: str = Query(default=""),
    model: str | None = Query(default=None),
    mode: str | None = Query(default="demo"),
    limit: int = Query(default=12, ge=1, le=30),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc
    try:
        results = index.search_tags(q, limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {exc.args[0]}") from exc
    return {"model": index.spec.as_dict(), "mode": mode or "demo", "results": results}


@app.get("/api/tags/random")
def random_tags(
    model: str | None = Query(default=None),
    mode: str | None = Query(default="demo"),
    limit: int = Query(default=1, ge=1, le=30),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc
    try:
        results = index.random_tags(limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {exc.args[0]}") from exc
    return {"model": index.spec.as_dict(), "mode": mode or "demo", "results": results}


@app.get("/api/tags/tracks")
def tracks_for_tag(
    tag: str = Query(...),
    model: str | None = Query(default=None),
    mode: str | None = Query(default="demo"),
    limit: int = Query(default=12, ge=1, le=30),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc
    try:
        results = index.tracks_for_tag(tag, limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {exc.args[0]}") from exc
    return {"model": index.spec.as_dict(), "mode": mode or "demo", "tag": tag, "results": results}


@app.get("/api/random")
def random_tracks(
    model: str | None = Query(default=None),
    mode: str | None = Query(default="demo"),
    limit: int = Query(default=12, ge=1, le=30),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc
    try:
        results = index.random(limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {exc.args[0]}") from exc
    return {"model": index.spec.as_dict(), "mode": mode or "demo", "results": results}


@app.get("/api/recommend/{spotify_id}")
def recommend(
    spotify_id: str,
    model: str | None = Query(default=None),
    mode: str | None = Query(default="demo"),
    space: str | None = Query(default=None),
    blend: float = Query(default=0.5, ge=0.0, le=1.0),
    limit: int = Query(default=10, ge=1, le=25),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc

    try:
        return index.recommend(spotify_id, limit, space=space, blend=blend, mode=mode)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown spotify_id: {spotify_id}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=f"Track not available for evaluation mode: {exc.args[0]}") from exc
    except ValueError as exc:
        if exc.args and exc.args[0] in {"demo", "evaluation"}:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {exc.args[0]}") from exc
        raise HTTPException(status_code=400, detail=f"Unknown recommendation space: {exc.args[0]}") from exc


@app.post("/api/recommend/upload")
async def recommend_uploaded_clip(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    space: str | None = Form(default=None),
    blend: float = Form(default=0.5),
    limit: int = Form(default=10),
    clip_start_sec: float = Form(default=0.0),
):
    if clip_start_sec < 0:
        raise HTTPException(status_code=400, detail="clip_start_sec must be non-negative.")
    if limit < 1 or limit > 25:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 25.")
    if blend < 0.0 or blend > 1.0:
        raise HTTPException(status_code=400, detail="blend must be between 0.0 and 1.0.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing upload filename.")

    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc

    suffix = Path(file.filename).suffix or ".wav"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            handle.write(await file.read())
        embedded = embed_uploaded_clip(index.spec, temp_path, clip_start_sec=clip_start_sec, filename=file.filename)
        payload = index.recommend_from_query_embeddings(
            embedded.query,
            embedded.embeddings,
            limit=limit,
            space=space,
            blend=blend,
        )
        payload["query"]["clip_duration_sec"] = AUDIO_CLIP_SECONDS
        return payload
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {exc}") from exc
    finally:
        await file.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.get("/api/artist-profile")
def artist_profile(
    artist: str = Query(...),
    model: str | None = Query(default=None),
    exclude_spotify_id: str | None = Query(default=None),
    limit: int = Query(default=8, ge=1, le=30),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc
    return index.artist_profile(artist=artist, exclude_spotify_id=exclude_spotify_id, limit=limit)


@app.get("/api/distance/{spotify_id_a}/{spotify_id_b}")
def distance(
    spotify_id_a: str,
    spotify_id_b: str,
    model: str | None = Query(default=None),
    space: str | None = Query(default=None),
    blend: float = Query(default=0.5, ge=0.0, le=1.0),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc

    try:
        return index.distance(spotify_id_a, spotify_id_b, space=space, blend=blend)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown spotify_id: {exc.args[0]}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown recommendation space: {exc.args[0]}") from exc


@app.get("/api/neighborhood/{spotify_id}")
def neighborhood(
    spotify_id: str,
    model: str | None = Query(default=None),
    space: str | None = Query(default=None),
    blend: float = Query(default=0.5, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=5, le=40),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc

    try:
        return index.neighborhood_map(spotify_id, space=space, blend=blend, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown spotify_id: {exc.args[0]}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown recommendation space: {exc.args[0]}") from exc


@app.get("/api/manifold")
def manifold(
    model: str | None = Query(default=None),
    spotify_id: str | None = Query(default=None),
    space: str | None = Query(default=None),
    blend: float = Query(default=0.5, ge=0.0, le=1.0),
    sample_limit: int = Query(default=220, ge=1, le=50000),
    neighbor_limit: int = Query(default=24, ge=1, le=200),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc

    try:
        return index.manifold_projection(
            spotify_id=spotify_id,
            space=space,
            blend=blend,
            sample_limit=sample_limit,
            neighbor_limit=neighbor_limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown spotify_id: {exc.args[0]}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown recommendation space: {exc.args[0]}") from exc


@app.get("/api/global-stats")
def global_stats(
    model: str | None = Query(default=None),
    space: str | None = Query(default=None),
    blend: float = Query(default=0.5, ge=0.0, le=1.0),
    pair_limit: int = Query(default=10, ge=3, le=25),
    song_limit: int = Query(default=10, ge=3, le=25),
):
    try:
        index = get_index(model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}") from exc

    try:
        return index.global_stats(space=space, blend=blend, pair_limit=pair_limit, song_limit=song_limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown recommendation space: {exc.args[0]}") from exc
