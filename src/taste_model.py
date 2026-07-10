"""Train a music-taste model on liked vs. candidate tracks — no audio features.

Labels
------
positive  : track is in Liked Songs or flagged is_saved
unlabeled : tracks from Spotify's personalized mixes that the user has NOT
            saved. These are treated as weak negatives (PU-learning setup):
            Spotify already thinks the user might like them, so real
            performance against true negatives would be HIGHER than reported.

Evaluation
----------
Leave-one-artist-out (grouped) CV: the model never sees the test artist in
training, so it cannot win by memorizing artist names — it must generalize
taste (genre, era, duration, title style).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.RandomState(42)


def load() -> pd.DataFrame:
    df = pd.DataFrame(json.loads((ROOT / "data" / "tracks_enriched.json").read_text()))

    # Overlay hand-curated enrichment where the API pass came up empty.
    knowledge_path = ROOT / "data" / "knowledge_enrichment.json"
    if knowledge_path.exists():
        knowledge = json.loads(knowledge_path.read_text())
        for i, uri in df["uri"].items():
            for field, value in knowledge.get(uri, {}).items():
                if field not in df.columns or pd.isna(df.at[i, field]):
                    df.at[i, field] = value

    df["label"] = (
        df["sources"].apply(lambda s: "liked_songs" in s) | df["is_saved"]
    ).astype(int)
    df["primary_artist"] = (
        df["artist"].fillna("unknown").str.split(",").str[0].str.strip().str.lower()
    )
    # Unknown-artist tracks must not collapse into one giant CV group:
    # give each its own group so grouped CV stays leave-one-ARTIST-out.
    df["cv_group"] = np.where(
        df["primary_artist"] == "unknown", df["uri"], df["primary_artist"]
    )
    # Co-occurrence across Spotify's mixes only — liked_songs defines the
    # label, so counting it here would leak.
    df["n_mix_sources"] = df["sources"].apply(
        lambda s: len([x for x in s if x not in ("liked_songs", "recently_played_samples")])
    )
    df["text"] = df["title"].fillna("") + " " + df["artist"].fillna("")
    return df


def build_features(df: pd.DataFrame):
    """Content features only.

    Deliberately EXCLUDED (both leak):
    - source counts: the label is defined by source membership (liked_songs
      vs. mix), so any source-derived feature encodes the label's origin.
    - artist liked-rate: under leave-one-artist-out CV the test artist's rate
      is computed from labels sitting in the test fold. It is added to the
      FINAL ranking model only (see main), never to the evaluated one.
    """
    genre = pd.get_dummies(df["genre"].fillna("unknown"), prefix="genre")
    year = df["release_year"].fillna(df["release_year"].median())
    duration = df["duration_ms"].fillna(df["duration_ms"].median()) / 1000
    explicit = df["explicit"].fillna(False).astype(int)

    dense = pd.DataFrame({"year": year, "duration_s": duration, "explicit": explicit})
    dense = pd.concat([dense, genre], axis=1).astype(float)
    # Columns with no data at all (e.g. release_year before enrichment) become 0.
    dense = dense.fillna(0.0)
    dense_scaled = StandardScaler().fit_transform(dense)

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=800)
    text = tfidf.fit_transform(df["text"].str.lower())

    X = hstack([dense_scaled, text]).tocsr()
    feature_names = list(dense.columns) + [f"tfidf:{t}" for t in tfidf.get_feature_names_out()]
    return X, dense, feature_names


def artist_liked_rate(df: pd.DataFrame) -> np.ndarray:
    """Leave-one-out artist liked-rate for the final ranking model only."""
    grp = df.groupby("primary_artist")["label"]
    sums, counts = grp.transform("sum"), grp.transform("count")
    prior = df["label"].mean()
    known = (df["primary_artist"] != "unknown") & (counts > 1)
    return np.where(known, (sums - df["label"]) / np.maximum(counts - 1, 1), prior)


def evaluate(X, y, groups) -> dict:
    """Grouped CV: pool out-of-fold scores, then compute one AUC."""
    models = {
        "logistic_regression": LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120, max_depth=3, learning_rate=0.08, random_state=42
        ),
    }
    results = {}
    logo = LeaveOneGroupOut()
    for name, model in models.items():
        oof = np.full(len(y), np.nan)
        for tr, te in logo.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            Xtr = X[tr].toarray() if name == "gradient_boosting" else X[tr]
            Xte = X[te].toarray() if name == "gradient_boosting" else X[te]
            model.fit(Xtr, y[tr])
            oof[te] = model.predict_proba(Xte)[:, 1]
        mask = ~np.isnan(oof)
        auc = roc_auc_score(y[mask], oof[mask])
        top10 = np.argsort(-oof[mask])[:10]
        results[name] = {
            "roc_auc": round(float(auc), 3),
            "precision_at_10": round(float(y[mask][top10].mean()), 2),
            "n_scored": int(mask.sum()),
        }
    return results


def explain(df_row, dense_row, weights, feature_names, k=3) -> str:
    contrib = dense_row * weights[: len(dense_row)]
    top = np.argsort(-np.abs(contrib))[:k]
    bits = []
    for i in top:
        name, w = feature_names[i], contrib[i]
        direction = "+" if w > 0 else "-"
        bits.append(f"{direction}{name.replace('genre_', '')}")
    return ", ".join(bits)


def main() -> None:
    df = load()
    X, dense, feature_names = build_features(df)
    y = df["label"].to_numpy()
    groups = df["cv_group"].to_numpy()

    print(f"{len(df)} unique tracks | {y.sum()} positives | {len(df) - y.sum()} weak negatives")
    print(f"{df['genre'].notna().sum()} tracks with genre | {df['primary_artist'].nunique()} artists\n")

    cv = evaluate(X, y, groups)
    for name, m in cv.items():
        print(f"{name}: leave-artist-out AUC={m['roc_auc']} | P@10={m['precision_at_10']}")

    # Final ranking model: fit LR on everything, score the unlabeled pool.
    # Here (and only here) add the leave-one-out artist liked-rate — appended
    # after TF-IDF so dense feature indices stay aligned for explanations.
    rate = artist_liked_rate(df).reshape(-1, 1)
    X_final = hstack([X, rate]).tocsr()
    final = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")
    final.fit(X_final, y)
    scores = final.predict_proba(X_final)[:, 1]
    weights = final.coef_[0]

    dense_scaled = StandardScaler().fit_transform(dense)
    pool = df[y == 0].copy()
    pool["score"] = scores[y == 0]
    pool = pool.sort_values("score", ascending=False).head(20)
    pool["why"] = [
        explain(row, dense_scaled[i], weights, feature_names)
        for i, row in zip(pool.index, pool.itertuples())
    ]

    preds = pool[["title", "artist", "genre", "release_year", "score", "why", "uri"]]
    preds["score"] = preds["score"].round(3)
    out = {
        "cv_metrics": cv,
        "caveats": [
            "PU learning: 'negatives' are unsaved tracks from Spotify's own personalized mixes — plausible likes, not true dislikes. Reported AUC is a lower bound.",
            "Tiny dataset (~170 tracks); metrics have wide confidence intervals.",
            "Candidate pool is pre-filtered by Spotify's recommender (popularity/selection bias).",
        ],
        "top_20": preds.to_dict(orient="records"),
    }
    (ROOT / "outputs" / "predictions.json").write_text(json.dumps(out, indent=1))

    lines = [
        "# You Will Love These — model-ranked picks\n",
        f"Leave-artist-out CV: LR AUC **{cv['logistic_regression']['roc_auc']}**, "
        f"GBT AUC **{cv['gradient_boosting']['roc_auc']}**\n",
        "| # | Track | Artist | Genre | Year | Score |",
        "|---|-------|--------|-------|------|-------|",
    ]
    for rank, r in enumerate(out["top_20"], 1):
        lines.append(
            f"| {rank} | {r['title']} | {r['artist'] or '?'} | {r['genre'] or '?'} "
            f"| {int(r['release_year']) if r['release_year'] else '?'} | {r['score']} |"
        )
    lines += ["", "## Caveats", *[f"- {c}" for c in out["caveats"]]]
    (ROOT / "outputs" / "report.md").write_text("\n".join(lines))
    print(f"\nWrote outputs/predictions.json and outputs/report.md")


if __name__ == "__main__":
    main()
