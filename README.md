# 🎵 Vibe Finder — A Song Recommender

Vibe Finder recommends songs that share the same musical "vibe" as a track you
already like. Instead of relying on genre labels, it groups songs by their
**audio features** (danceability, energy, valence, tempo, and more) and suggests
the closest matches within the same cluster.

**🔗 Live app:** _add your Streamlit link here after deploying_
**📊 Presentation:** _add your slides link here_

---

## What it does

- Type a song you like → get 5 recommendations with a similar sound
- Browse songs by **genre** and **artist**
- See *why* two songs match, with a feature comparison chart
- Build and manage your own playlists, with a Spotify play button for each track

---

## How it works

1. **Scraping** — the current Billboard Hot 100 is scraped from the web.
2. **Data collection** — for each song, audio features are gathered through the
   Deezer and ReccoBeats public APIs, producing a dataset of ~1,365 tracks.
3. **Clustering** — features are scaled, then **K-Means** groups the songs into
   **7 clusters**. The internal metrics (elbow, silhouette, Calinski-Harabasz,
   Davies-Bouldin) don't point to a single "best" number of clusters, so k = 7
   was chosen as a practical balance and validated with the hit-rate test below.
4. **Recommending** — for a chosen song, the app finds the nearest tracks
   *inside its own cluster* using distances in the feature space.

### Does it actually work?

Yes. Measured as the share of recommendations that share the genre of the
original song (see `notebooks/gnod_evaluation.ipynb`):

| Method | Same-genre hit rate |
|---|---|
| Random pick (whole library) | ~8% |
| Closest songs in the same cluster (the app) | ~39% |

The app's recommendations are about **5× better than random** — evidence that
grouping songs by sound really does capture musical similarity.

---

## Tech stack

- **Python**, **pandas**, **scikit-learn** (StandardScaler, K-Means, metrics)
- **requests**, **BeautifulSoup** (web scraping)
- **Deezer** & **ReccoBeats** APIs (audio features)
- **Streamlit** (web app)

---

## Project structure

```
.
├── app.py                        # the Streamlit app
├── requirements.txt
├── data/                         # datasets
│   └── gnod_songs_clustered.csv
└── notebooks/
    ├── gnod_scraping_billboard.ipynb
    ├── gnod_data_collection.ipynb
    ├── gnod_clustering.ipynb
    └── gnod_evaluation.ipynb
```

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Pariya**
