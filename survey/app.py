"""
Music Recommendation Survey
----------------------------
Run with:  streamlit run app.py

Expects survey_data.json in the same directory.
Results are saved to results/<name>_<timestamp>.json
"""

import json
import random
from datetime import datetime
from pathlib import Path

import streamlit as st

SURVEY_DATA_PATH = Path(__file__).parent / "survey_data.json"
RESULTS_DIR      = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STYLES = """
<style>
    .track-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .track-title  { font-size: 1rem; font-weight: 700; color: #cdd6f4; margin: 0 0 2px 0; }
    .track-artist { font-size: 0.85rem; color: #a6adc8; margin: 0; }
    .playlist-header {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #cba6f7;
        padding: 8px 0 12px 0;
    }
    .query-box {
        background: #181825;
        border-left: 4px solid #cba6f7;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
    }
    .query-label { font-size: 0.75rem; text-transform: uppercase;
                   letter-spacing: 2px; color: #cba6f7; margin: 0 0 4px 0; }
    .query-title  { font-size: 1.5rem; font-weight: 800; color: #cdd6f4; margin: 0; }
    .query-artist { font-size: 1rem; color: #a6adc8; margin: 4px 0 0 0; }
</style>
"""


@st.cache_data
def load_survey_data():
    with open(SURVEY_DATA_PATH) as f:
        return json.load(f)


def save_results(name: str, data: dict) -> Path:
    result = {
        "respondent": name,
        "timestamp":  datetime.now().isoformat(),
        "ratings":    [],
    }
    for q_idx, query_data in enumerate(data["queries"]):
        query_result = {
            "query_track": query_data["query"],
            "playlists":   [],
        }
        order = st.session_state.playlist_order[q_idx]
        for rank, playlist_idx in enumerate(order):
            playlist = query_data["playlists"][playlist_idx]
            label    = chr(65 + rank)
            query_result["playlists"].append({
                "label":         label,
                "model":         playlist["model"],
                "track_ratings": st.session_state.ratings[q_idx][label],
            })
        result["ratings"].append(query_result)

    filename = RESULTS_DIR / f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    return filename


def init_session(data: dict):
    if "initialized" in st.session_state:
        return
    st.session_state.playlist_order = {}
    for i, q in enumerate(data["queries"]):
        order = list(range(len(q["playlists"])))
        random.shuffle(order)
        st.session_state.playlist_order[i] = order
    st.session_state.query_idx  = 0
    st.session_state.ratings    = {}
    st.session_state.initialized = True


def render_track(track: dict, q_idx: int, label: str, t_idx: int) -> str | None:
    """Render a single track card with optional audio preview. Returns the rating."""
    preview = track.get("preview_url")
    year    = track.get("year", "?")

    st.markdown(
        f"""<div class="track-card">
              <p class="track-title">{track['name']}</p>
              <p class="track-artist">{track['artist']} &nbsp;·&nbsp; {year}</p>
            </div>""",
        unsafe_allow_html=True,
    )
    if preview:
        st.audio(preview, format="audio/mpeg")
    else:
        st.caption("No preview available")

    rating = st.radio(
        label="rating",
        options=["👍 Good", "😐 Neutral", "👎 Bad"],
        key=f"q{q_idx}_p{label}_t{t_idx}",
        index=None,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.write("")
    return rating


def main():
    st.set_page_config(page_title="Music Recommendation Survey", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)

    # ── Welcome ──────────────────────────────────────────────────────────────────
    if "name" not in st.session_state:
        st.title("🎵 Music Recommendation Survey")
        st.markdown(
            "You'll be shown **10 songs**. For each song, three playlists of "
            "5 recommendations are shown — listen to the previews and rate each "
            "track **Good**, **Neutral**, or **Bad**.\n\n"
            "The playlists are unlabelled — just go with your gut!"
        )
        st.divider()
        name = st.text_input("Your name:")
        if st.button("Start Survey", disabled=not name.strip(), type="primary"):
            st.session_state.name = name.strip()
            st.rerun()
        return

    data      = load_survey_data()
    init_session(data)
    queries   = data["queries"]
    n_queries = len(queries)
    q_idx     = st.session_state.query_idx

    # ── Complete ──────────────────────────────────────────────────────────────────
    if q_idx >= n_queries:
        st.title("🎉 Survey Complete!")
        st.write(f"Thank you, **{st.session_state.name}**! All {n_queries} query tracks rated.")
        if st.button("Save & Download Results", type="primary"):
            path = save_results(st.session_state.name, data)
            with open(path) as f:
                st.download_button(
                    label="⬇️ Download results JSON",
                    data=f.read(),
                    file_name=path.name,
                    mime="application/json",
                )
            st.balloons()
        return

    # ── Survey page ───────────────────────────────────────────────────────────────
    st.progress(q_idx / n_queries, text=f"Song {q_idx + 1} of {n_queries}")

    query_data = queries[q_idx]
    q          = query_data["query"]

    # Query track box
    preview = q.get("preview_url")
    st.markdown(
        f"""<div class="query-box">
              <p class="query-label">Query Track</p>
              <p class="query-title">{q['name']}</p>
              <p class="query-artist">{q['artist']} &nbsp;·&nbsp; {q.get('year', '?')}</p>
            </div>""",
        unsafe_allow_html=True,
    )
    if preview:
        st.audio(preview, format="audio/mpeg")
    st.divider()

    # Initialise ratings for this query
    if q_idx not in st.session_state.ratings:
        st.session_state.ratings[q_idx] = {}

    order       = st.session_state.playlist_order[q_idx]
    n_playlists = len(order)
    cols        = st.columns(n_playlists, gap="large")
    all_rated   = True

    for col, rank, playlist_idx in zip(cols, range(n_playlists), order):
        label    = chr(65 + rank)
        playlist = query_data["playlists"][playlist_idx]

        if label not in st.session_state.ratings[q_idx]:
            st.session_state.ratings[q_idx][label] = [None] * len(playlist["tracks"])

        with col:
            st.markdown(f'<div class="playlist-header">Playlist {label}</div>', unsafe_allow_html=True)
            for t_idx, track in enumerate(playlist["tracks"]):
                rating = render_track(track, q_idx, label, t_idx)
                st.session_state.ratings[q_idx][label][t_idx] = rating
                if rating is None:
                    all_rated = False

    # ── Navigation ────────────────────────────────────────────────────────────────
    st.divider()
    if not all_rated:
        st.warning("Please rate all tracks before continuing.")

    nav_l, _, nav_r = st.columns([1, 6, 1])
    with nav_l:
        if q_idx > 0 and st.button("← Back"):
            st.session_state.query_idx -= 1
            st.rerun()
    with nav_r:
        btn_label = "Next →" if q_idx < n_queries - 1 else "Finish ✓"
        if st.button(btn_label, disabled=not all_rated, type="primary"):
            st.session_state.query_idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
