# spotify-taste-model

Predicting which songs I'll like from my real Spotify library — **without audio features**
(Spotify deprecated the audio-features endpoint in Nov 2024), on ~150 tracks, with an
evaluation designed to be hard to fool.

Built as a one-day experiment on my own listening data, pulled live through the Spotify MCP
connector. The interesting part isn't the model — it's the three evaluation traps this dataset
sets, and how each one showed up in the metrics before being fixed.

## Problem setup

| | |
|---|---|
| **Positives (51)** | My Liked Songs + tracks flagged saved |
| **Weak negatives (97)** | Tracks from my four Spotify "Made For You" mixes that I *haven't* saved |
| **Features** | genre, release year, duration, explicit flag, char-ngram TF-IDF over title+artist |
| **Models** | L2 logistic regression vs. gradient-boosted trees (sklearn) |
| **Evaluation** | leave-one-artist-out grouped CV, ROC-AUC + precision@10 |

This is a **PU-learning** problem (positive + unlabeled), not true binary classification:
the "negatives" are songs Spotify's recommender already picked for me — plausible likes, not
dislikes. All reported metrics are therefore *lower bounds* on real discrimination.

## The three traps (a story in AUC)

| Run | Leave-artist-out AUC | What was happening |
|-----|---------------------|--------------------|
| v1: no enrichment | 0.28–0.47 | 100+ tracks had no artist/genre; unknown artists collapsed into one giant CV group. Worse than coin-flip = the pipeline, not the taste, was broken. |
| v2: enriched, naive features | **0.95** | Looked amazing. Was leakage: (a) a source-count feature encoded the label's origin (liked songs come *from* the Liked Songs source); (b) artist liked-rate was computed dataset-wide, so test folds carried their own labels. |
| v3: leak-free | **0.615** (P@10 = 0.6) | Honest. Content features only under CV; the model must predict taste for artists it has never seen. |

If you only remember one thing: **a 0.95 on 148 rows is an accusation, not a result.**

## Why leave-one-artist-out?

Random splits let the model win by memorizing artists (every Coldplay track I've liked
predicts the next one). Grouped CV holds out entire artists, forcing generalization to
genre/era/duration/title-style. Tracks with unresolvable artists get singleton groups so
they can't form one mega-group.

## Enrichment

`src/enrich.py` recovers artist/genre/release-year from the free iTunes Search API
(throttled, cached, fuzzy title matching with confidence scores). The committed
`data/knowledge_enrichment.json` is a hand-curated fallback used when this ran in a
network-restricted sandbox — high-confidence entries only, unknowns stay NaN (~64% genre
coverage).

## Results

Top-ranked unliked tracks (final model adds a leave-one-out artist liked-rate, excluded
from the evaluated model): Feel Good Inc., Nikes on My Feet, Tongue Tied, I Wanna Be Yours,
Riptide — see `outputs/predictions.json` / `outputs/report.md`.

Favorite failure mode: a song literally titled *"A COLD PLAY"* ranked #4, because char-ngram
TF-IDF learned that things spelled like "coldplay" are things I like. Text features on titles
are a real signal and a real trap.

## Run it

```bash
pip install pandas scikit-learn scipy requests
python3 src/enrich.py       # optional; needs network. Cached in data/
python3 src/taste_model.py  # trains, evaluates, writes outputs/
```

## Limitations & next steps

- 148 tracks; every metric has wide error bars. More library paging → better.
- Candidate pool is pre-filtered by Spotify's recommender (popularity/selection bias);
  the model reranks Spotify's shortlist rather than searching the open catalog.
- No listening-frequency signal (play counts would beat the binary saved-flag).
- Next: score an out-of-mix candidate pool, A/B my reranking vs. Spotify's mix order
  (blind test), and add artist-level embeddings from co-occurrence across playlists.

*Data collected from my own Spotify account via the Spotify MCP connector; ideation
brainstormed with GPT-5 Codex; built with Claude Code.*
