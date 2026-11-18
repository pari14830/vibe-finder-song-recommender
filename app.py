import html

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

st.set_page_config(
    page_title="Vibe Finder",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One palette used everywhere: navy, sea, mist, coral.
INK = "#1D3557"
SEA = "#457B9D"
MIST = "#A8DADC"
CORAL = "#E63946"
PAPER = "#F4F7FB"

st.markdown(
    f"""
    <style>
      .stApp {{
        background:
          radial-gradient(900px 420px at 0% -10%, rgba(168, 218, 220, 0.45), transparent 60%),
          radial-gradient(700px 380px at 100% 0%, rgba(230, 57, 70, 0.10), transparent 55%),
          linear-gradient(180deg, #F7FBFC 0%, {PAPER} 40%, #EEF2F7 100%);
        color: {INK};
      }}
      [data-testid="stHeader"] {{
        background: transparent;
      }}
      [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1D3557 0%, #152A45 100%);
        border-right: 0;
      }}
      [data-testid="stSidebar"] * {{
        color: #F7FBFC !important;
      }}
      [data-testid="stSidebar"] [data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(168, 218, 220, 0.25);
        border-radius: 14px;
        padding: 10px 12px;
      }}
      [data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
        color: {MIST} !important;
      }}
      .hero {{
        background: linear-gradient(120deg, #1D3557 0%, #245074 55%, #1D3557 100%);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(29, 53, 87, 0.18);
      }}
      .hero h1 {{
        color: white;
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.03em;
      }}
      .hero p {{
        color: {MIST};
        font-size: 1.05rem;
        margin: 8px 0 0 0;
      }}
      .pill {{
        display: inline-block;
        background: {CORAL};
        color: white;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        margin-left: 8px;
        letter-spacing: 0.02em;
      }}
      .song-card {{
        background: white;
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(29, 53, 87, 0.08);
        border-left: 5px solid {CORAL};
        margin-bottom: 8px;
        box-shadow: 0 6px 16px rgba(29, 53, 87, 0.05);
      }}
      .song-title {{
        font-size: 17px;
        font-weight: 700;
        color: {INK};
      }}
      .song-artist {{
        color: {SEA};
      }}
      .song-link {{
        color: {INK};
        text-decoration: none;
        font-size: 14px;
        font-weight: 600;
      }}
      .playlist-head {{
        background: {INK};
        padding: 12px 18px;
        border-radius: 12px;
        margin-top: 8px;
      }}
      .playlist-head span {{
        color: white;
        font-size: 18px;
        font-weight: 700;
      }}
      .playlist-head small {{
        color: {MIST};
        font-size: 14px;
        margin-left: 8px;
      }}
      .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
      }}
      .stTabs [data-baseweb="tab"] {{
        background: white;
        border-radius: 999px;
        padding: 8px 16px;
        border: 1px solid rgba(29, 53, 87, 0.08);
      }}
      .stTabs [aria-selected="true"] {{
        background: {INK} !important;
        color: white !important;
      }}
      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: white;
        border-radius: 16px;
        border-color: rgba(29, 53, 87, 0.08);
        box-shadow: 0 8px 20px rgba(29, 53, 87, 0.04);
      }}
      .stButton > button {{
        border-radius: 999px;
        font-weight: 600;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎵 Vibe Finder</h1>
        <p>Give me a song you love, and I'll find others that feel the same.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# The 9 audio features the clusters were built on.
FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
# For the "why they match" chart, 0-1 features only (loudness and tempo have a
# very different scale and would flatten the chart).
CHART_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]
PAGE_SIZE = 6


@st.cache_data
def load_data():
    return pd.read_csv("data/gnod_songs_clustered.csv")


@st.cache_data
def build_distances(frame):
    """Scaled features + a song-to-song distance matrix, used to pick the closest songs."""
    scaled = StandardScaler().fit_transform(frame[FEATURES])
    distances = pairwise_distances(scaled)
    return distances


df = load_data()
D = build_distances(df)
cluster_profile = df.groupby("cluster")[FEATURES].mean()
overall_mean = df[FEATURES].mean()
overall_std = df[FEATURES].std()


def describe_cluster(cluster_id):
    row = cluster_profile.loc[cluster_id]
    # distance from the overall average measured in standard deviations, so
    # features on different scales (tempo in BPM, loudness in dB) compare fairly
    diff = (row - overall_mean) / overall_std
    return diff.sort_values(ascending=False).head(2).index.tolist()


def recommend(i, n=5, seed=0):
    """Recommend n songs closest to song at row index i, from its own cluster."""
    song_cluster = df.loc[i, "cluster"]

    # positions of the other songs in the same cluster
    same_cluster = df.index[(df["cluster"] == song_cluster) & (df.index != i)].to_numpy()
    if len(same_cluster) == 0:
        return df.loc[[]], song_cluster

    # order them by distance to the chosen song, closest first
    order = np.argsort(D[i, same_cluster])
    closest = same_cluster[order]

    # take a few more than n, then shuffle those so "New matches" varies a little
    pool = closest[: max(n * 3, n)]
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(pool)[:n]

    results = df.loc[chosen, ["song", "artist", "genre", "id"]]
    return results, song_cluster


def song_card_html(row):
    link = "https://open.spotify.com/track/" + str(row["id"])
    return (
        '<div class="song-card">'
        f'<span class="song-title">{html.escape(str(row["song"]))}</span>'
        f'<span class="pill">{html.escape(str(row["genre"]))}</span><br>'
        f'<span class="song-artist">{html.escape(str(row["artist"]))}</span><br>'
        f'<a class="song-link" href="{link}" target="_blank">Open in Spotify</a>'
        "</div>"
    )


def record_feedback(key, value):
    previous = st.session_state.seen_feedback.get(key)
    if value is None or value == previous:
        return
    if previous == 1:
        st.session_state.likes -= 1
    elif previous == 0:
        st.session_state.dislikes -= 1
    if value == 1:
        st.session_state.likes += 1
    elif value == 0:
        st.session_state.dislikes += 1
    st.session_state.seen_feedback[key] = value


def show_song_card(row, key_prefix, show_player=False, show_feedback=False):
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(song_card_html(row), unsafe_allow_html=True)
            if show_player:
                # one shared player (rendered at the top) — so only one song
                # plays at a time instead of every card playing at once
                if st.button("▶ Play", key="play_" + key_prefix + row["id"]):
                    st.session_state.now_playing = row["id"]
                    st.rerun()
        with right:
            with st.popover("Save"):
                playlist_names = list(st.session_state.playlists.keys())
                new_song = {"song": row["song"], "artist": row["artist"], "id": row["id"]}
                if playlist_names:
                    chosen = st.selectbox(
                        "Playlist",
                        playlist_names,
                        key="pl_" + key_prefix + row["id"],
                    )
                    if st.button("Add", key="add_" + key_prefix + row["id"], type="primary"):
                        if new_song not in st.session_state.playlists[chosen]:
                            st.session_state.playlists[chosen].append(new_song)
                        st.success("Added to " + chosen)
                with st.form("form_" + key_prefix + row["id"], clear_on_submit=True):
                    new_pl = st.text_input("New playlist")
                    submitted = st.form_submit_button("Create and add")
                if submitted:
                    cleaned = new_pl.strip()
                    if cleaned:
                        st.session_state.playlists.setdefault(cleaned, [])
                        if new_song not in st.session_state.playlists[cleaned]:
                            st.session_state.playlists[cleaned].append(new_song)
                        st.success("Saved to " + cleaned)
                    else:
                        st.warning("Give the playlist a name first.")
            if show_feedback:
                st.caption("Good match?")
                vote = st.feedback("thumbs", key="fb_" + key_prefix + row["id"])
                record_feedback(key_prefix + row["id"], vote)


def show_page(frame, key_prefix, limit_key, show_player=False, show_feedback=False):
    limit = st.session_state[limit_key]
    visible = frame.head(limit)
    for _, row in visible.iterrows():
        show_song_card(row, key_prefix, show_player=show_player, show_feedback=show_feedback)
    remaining = len(frame) - len(visible)
    if remaining > 0 and st.button(f"Show {min(PAGE_SIZE, remaining)} more", key="more_" + limit_key):
        st.session_state[limit_key] += PAGE_SIZE
        st.rerun()


if "playlists" not in st.session_state:
    st.session_state.playlists = {}
if "results" not in st.session_state:
    st.session_state.results = None
if "chosen_song" not in st.session_state:
    st.session_state.chosen_song = None
if "match_seed" not in st.session_state:
    st.session_state.match_seed = 1
if "likes" not in st.session_state:
    st.session_state.likes = 0
if "dislikes" not in st.session_state:
    st.session_state.dislikes = 0
if "seen_feedback" not in st.session_state:
    st.session_state.seen_feedback = {}
if "now_playing" not in st.session_state:
    st.session_state.now_playing = None
if "browse_limit" not in st.session_state:
    st.session_state.browse_limit = PAGE_SIZE
if "popular_limit" not in st.session_state:
    st.session_state.popular_limit = PAGE_SIZE

saved_count = sum(len(songs) for songs in st.session_state.playlists.values())

with st.sidebar:
    st.markdown("### Vibe Finder")
    st.caption("Songs matched by how they sound, not only by genre.")
    st.metric("Songs", f"{len(df):,}")
    st.metric("Vibes", int(df["cluster"].nunique()))
    st.metric("Genres", int(df["genre"].nunique()))
    st.metric("Saved", saved_count)
    votes = st.session_state.likes + st.session_state.dislikes
    if votes:
        st.progress(st.session_state.likes / votes, text=f"{st.session_state.likes} likes · {st.session_state.dislikes} passes")
    st.caption("Clusters come from Spotify audio features: energy, danceability, valence, and more.")

# One shared player for the whole app: only one song ever plays at a time.
# Pressing "Play" on any card sets this and the single player below swaps track.
if st.session_state.now_playing:
    st.caption("▶ Now playing")
    st.components.v1.iframe(
        "https://open.spotify.com/embed/track/" + str(st.session_state.now_playing),
        height=80,
    )

tab_discover, tab_browse, tab_playlists, tab_popular = st.tabs(
    ["Discover", "Browse", "Playlists", "Popular"]
)

with tab_discover:
    # label each song with its artist so duplicate titles (e.g. two "Stand By Me")
    # can be told apart in the dropdown and point to the right song
    song_labels = df["song"] + " — " + df["artist"]
    label_to_index = {}
    for idx, label in song_labels.items():
        label_to_index.setdefault(label, idx)  # keep the first song with this label
    all_labels = sorted(label_to_index)
    if "discover_song" not in st.session_state:
        st.session_state.discover_song = all_labels[0]

    controls, actions = st.columns([3, 1])
    with actions:
        st.write("")
        st.write("")
        refresh = st.button("New matches", type="primary", use_container_width=True)
        surprise = st.button("Surprise me", use_container_width=True)

    if surprise:
        st.session_state.discover_song = song_labels.loc[df.sample(1).index[0]]
        st.session_state.match_seed += 1
        st.rerun()

    with controls:
        song_input = st.selectbox("Start with a song you like", all_labels, key="discover_song")
        how_many = st.slider("How many matches", 3, 8, 5)

    need_new = (
        refresh
        or surprise
        or st.session_state.chosen_song != song_input
        or st.session_state.results is None
        or len(st.session_state.results) != how_many
    )
    if need_new:
        if refresh or surprise:
            st.session_state.match_seed += 1
        i = label_to_index[song_input]
        results, cluster_id = recommend(i, how_many, st.session_state.match_seed)
        st.session_state.results = results
        st.session_state.chosen_song = song_input
        st.session_state.chosen_index = i
        st.session_state.cluster_id = cluster_id

    if st.session_state.results is None or len(st.session_state.results) == 0:
        st.warning("No other songs share this vibe yet. Try another song.")
    else:
        cluster_id = st.session_state.cluster_id
        top_features = describe_cluster(cluster_id)
        st.info(
            "These songs sit together because they are high in **"
            + top_features[0]
            + "** and **"
            + top_features[1]
            + "**."
        )

        chosen_row = df.loc[st.session_state.chosen_index]
        compare = pd.DataFrame(
            {
                "Your song": chosen_row[CHART_FEATURES].astype(float),
                "This vibe": cluster_profile.loc[cluster_id][CHART_FEATURES].astype(float),
            }
        )
        with st.expander("See why they match", expanded=False):
            st.bar_chart(compare, height=220)
            st.caption("Bars closer together mean the match shares that part of the sound.")

        for _, row in st.session_state.results.iterrows():
            show_song_card(row, "discover_", show_player=True, show_feedback=True)

with tab_browse:
    search_col, genre_col, artist_col = st.columns([2, 1, 1])
    with search_col:
        query = st.text_input("Search title or artist", placeholder="Try: Torn")
    genres = ["All"] + sorted(df["genre"].unique())
    with genre_col:
        chosen_genre = st.selectbox("Genre", genres, key="browse_genre")
    browse = df if chosen_genre == "All" else df[df["genre"] == chosen_genre]
    artists = ["All"] + sorted(browse["artist"].unique())
    if st.session_state.get("browse_artist") not in artists:
        st.session_state.browse_artist = "All"
    with artist_col:
        chosen_artist = st.selectbox("Artist", artists, key="browse_artist")
    if chosen_artist != "All":
        browse = browse[browse["artist"] == chosen_artist]

    energy_range, dance_range = st.columns(2)
    with energy_range:
        energy = st.slider("Energy", 0.0, 1.0, (0.0, 1.0), key="energy_filter")
    with dance_range:
        dance = st.slider("Danceability", 0.0, 1.0, (0.0, 1.0), key="dance_filter")

    browse = browse[
        browse["energy"].between(energy[0], energy[1])
        & browse["danceability"].between(dance[0], dance[1])
    ]
    if query:
        mask = browse["song"].str.contains(query, case=False, na=False, regex=False) | browse[
            "artist"
        ].str.contains(query, case=False, na=False, regex=False)
        browse = browse[mask]

    filter_key = f"{query}|{chosen_genre}|{chosen_artist}|{energy}|{dance}"
    if st.session_state.get("browse_filter") != filter_key:
        st.session_state.browse_filter = filter_key
        st.session_state.browse_limit = PAGE_SIZE

    match_count = len(browse)
    match_word = "song" if match_count == 1 else "songs"
    st.caption(f"{match_count:,} {match_word} match")
    if len(browse) == 0:
        st.info("Nothing fits those filters. Widen energy or danceability, or clear the search.")
    else:
        show_page(browse, "browse_", "browse_limit")

with tab_playlists:
    with st.container(border=True):
        with st.form("create_playlist_form", clear_on_submit=True):
            name_col, button_col = st.columns([3, 1])
            with name_col:
                new_name = st.text_input("Playlist name", placeholder="Late night drive")
            with button_col:
                st.write("")
                st.write("")
                create = st.form_submit_button("Create playlist", type="primary", use_container_width=True)
        if create:
            cleaned = new_name.strip()
            if not cleaned:
                st.warning("Give the playlist a name first.")
            elif cleaned in st.session_state.playlists:
                st.warning("You already have a playlist with that name.")
            else:
                st.session_state.playlists[cleaned] = []
                st.success(f"Created {cleaned}. Add songs from Discover or Browse.")

    if not st.session_state.playlists:
        st.info("No playlists yet. Create one, then save songs from the other tabs.")
    else:
        for name in list(st.session_state.playlists.keys()):
            songs = st.session_state.playlists[name]
            st.markdown(
                '<div class="playlist-head"><span>'
                + html.escape(name)
                + "</span><small>"
                + str(len(songs))
                + (" song" if len(songs) == 1 else " songs")
                + "</small></div>",
                unsafe_allow_html=True,
            )
            if not songs:
                st.caption("Empty for now. Use Save on any song card.")
            for i, item in enumerate(list(songs)):
                link = "https://open.spotify.com/track/" + item["id"]
                song_col, remove_col = st.columns([5, 1])
                with song_col:
                    st.markdown(
                        f'**{item["song"]}** · {item["artist"]}  ·  [Spotify]({link})'
                    )
                with remove_col:
                    if st.button("Remove", key=f"rm_{name}_{i}"):
                        songs.pop(i)
                        st.rerun()
            if st.button(f"Delete {name}", key="del_" + name):
                del st.session_state.playlists[name]
                st.rerun()

with tab_popular:
    popular = df[df["is_hot100"] == True].copy()
    hot_genres = ["All"] + sorted(popular["genre"].unique())
    hot_genre = st.selectbox("Hot 100 genre", hot_genres)
    if hot_genre != "All":
        popular = popular[popular["genre"] == hot_genre]
    if st.session_state.get("popular_filter") != hot_genre:
        st.session_state.popular_filter = hot_genre
        st.session_state.popular_limit = PAGE_SIZE

    hot_count = len(popular)
    hot_word = "song" if hot_count == 1 else "songs"
    st.caption(f"{hot_count:,} {hot_word} from the Billboard Hot 100 in this library")
    if len(popular) == 0:
        st.info("No Hot 100 songs in that genre.")
    else:
        show_page(popular, "popular_", "popular_limit", show_player=True)