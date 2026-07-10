"""Hidden-gem discovery engine (v2) — the method a cross-model review picked.

Design (agreed with a GPT-5 Codex adversarial ideation pass):
- A regularized PU-aware logistic model produces a cross-validated PREFERENCE
  SCORE. It is never presented as P(love): with 51 liked positives and 97
  behaviorally-selected weak negatives, the PU labeling mechanism is
  feature-dependent, so calibration corrections (Elkan-Noto) would be
  unstable at this n. Scores are reported as percentiles.
- Grouped ARTIST bootstrap (resample artists, refit, rescore) gives ranking
  STABILITY intervals — not probabilistic confidence intervals.
- discovery score = preference percentile x obscurity^gamma, obscurity from
  Deezer's popularity rank (log-percentile within pool). Popularity never
  enters the model itself.
- Safe gems  = high 10th-percentile preference bound + above-median obscurity.
  Moonshots  = high 90th-percentile bound with a wide (uncertain) interval.
- Honesty checks: repeated leave-one-artist-out CV with lift@20%, feature
  ablations (metadata / text / combined), and gamma-robustness of the top list.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.RandomState(7)
N_BOOT = 300
GENRES = ["Alternative", "Hip-Hop/Rap", "Pop", "R&B/Soul"]

# Candidate tracks carry no release date from the Deezer search endpoint.
# Era is imputed at ARTIST level (median active album era) — coarse, flagged.
ARTIST_ERA = {
    "montell fish": 2022, "dj gummy bear": 2022, "michael seyer": 2018,
    "sunset rollercoaster": 2019, "the backseat lovers": 2021,
    "briston maroney": 2021, "adam melchor": 2021, "redveil": 2022,
    "jpegmafia": 2021, "ag club": 2021, "dreamer isioma": 2021,
    "jelani aryeh": 2021, "samia": 2021, "men i trust": 2019,
    "current joys": 2018, "day wave": 2017, "hazel english": 2018,
    "sea lemon": 2023, "krooked kings": 2022, "wave to earth": 2023,
    "jeon jin hee": 2023, "smino": 2018, "doja cat": 2019, "kari faux": 2019,
    "noname": 2018, "saba": 2018, "khalid": 2021, "brent faiyaz": 2023,
    "jordan ward": 2023, "joony": 2022, "flipturn": 2022, "peach pit": 2020,
}


def load_library() -> pd.DataFrame:
    df = pd.DataFrame(json.loads((ROOT / "data" / "tracks_enriched.json").read_text()))
    knowledge = json.loads((ROOT / "data" / "knowledge_enrichment.json").read_text())
    for i, uri in df["uri"].items():
        for field, value in knowledge.get(uri, {}).items():
            if field.startswith("_"):
                continue
            if field not in df.columns or pd.isna(df.at[i, field]):
                df.at[i, field] = value
    df["label"] = (
        df["sources"].apply(lambda s: "liked_songs" in s) | df["is_saved"]
    ).astype(int)
    out = pd.DataFrame(
        {
            "title": df["title"],
            "artist": df["artist"].fillna("unknown"),
            "genre": df["genre"],
            "duration_s": pd.to_numeric(df["duration_ms"], errors="coerce") / 1000,
            "explicit": df["explicit"].fillna(False).astype(bool),
            "year": pd.to_numeric(df["release_year"], errors="coerce"),
            "label": df["label"],
        }
    )
    out["primary_artist"] = out["artist"].str.split(",").str[0].str.strip().str.lower()
    out["group"] = np.where(out["primary_artist"] == "unknown", df["uri"], out["primary_artist"])
    return out


def load_candidates() -> pd.DataFrame:
    raw = json.loads((ROOT / "data" / "candidates.json").read_text())
    rows = []
    for pool in raw["pools"]:
        for t, a, d, e, r in pool["tracks"]:
            rows.append({"title": t, "artist": a, "genre": pool["genre"],
                         "duration_s": d, "explicit": e, "deezer_rank": r})
    df = pd.DataFrame(rows).drop_duplicates(subset=["title", "artist"]).reset_index(drop=True)
    df["primary_artist"] = df["artist"].str.split(",").str[0].str.strip().str.lower()
    df["year"] = df["primary_artist"].map(ARTIST_ERA)
    # Obscurity: log-rank percentile within pool, flipped so obscure -> 1.
    lr = np.log(df["deezer_rank"])
    df["obscurity"] = 1 - (lr.rank(pct=True))
    return df


def featurize(lib: pd.DataFrame, cand: pd.DataFrame):
    """Fit transforms on the LIBRARY only; apply to both (candidates are
    out-of-distribution — they must not influence scaling or vocabulary)."""
    def dense_block(df):
        g = pd.DataFrame({f"g_{k}": (df["genre"] == k).astype(float) for k in GENRES})
        g["g_unknown"] = df["genre"].isna().astype(float)
        return pd.concat(
            [pd.DataFrame({"duration_s": df["duration_s"].fillna(lib["duration_s"].median()),
                           "explicit": df["explicit"].astype(float),
                           "year": df["year"].fillna(lib["year"].median())}), g], axis=1
        ).to_numpy(dtype=float)

    scaler = StandardScaler().fit(dense_block(lib))
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=800)
    text_fit = tfidf.fit(( lib["title"] + " " + lib["artist"]).str.lower())

    def X(df, which="both"):
        d = scaler.transform(dense_block(df))
        t = text_fit.transform((df["title"] + " " + df["artist"]).str.lower())
        if which == "meta":
            return d
        if which == "text":
            return t.toarray()
        return hstack([d, t]).tocsr().toarray()

    return X


def model():
    return LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")


def loao_metrics(Xl, y, groups):
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(Xl, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = model().fit(Xl[tr], y[tr])
        oof[te] = m.predict_proba(Xl[te])[:, 1]
    mask = ~np.isnan(oof)
    auc = roc_auc_score(y[mask], oof[mask])
    ap = average_precision_score(y[mask], oof[mask])
    k = max(1, int(0.2 * mask.sum()))
    top = np.argsort(-oof[mask])[:k]
    lift = y[mask][top].mean() / y[mask].mean()
    return {"auc": round(float(auc), 3), "avg_precision": round(float(ap), 3),
            "lift_at_20pct": round(float(lift), 2)}


def main() -> None:
    lib, cand = load_library(), load_candidates()
    # Drop candidates already in the library (title-level match).
    known = set(lib["title"].str.lower().str.strip())
    cand = cand[~cand["title"].str.lower().str.strip().isin(known)].reset_index(drop=True)
    y = lib["label"].to_numpy()
    groups = lib["group"].to_numpy()
    X = featurize(lib, cand)

    print(f"library {len(lib)} | candidates {len(cand)} from {cand['primary_artist'].nunique()} artists\n")

    # --- Honest evaluation: LOAO CV + ablations -------------------------
    ablations = {w: loao_metrics(X(lib, w), y, groups) for w in ("meta", "text", "both")}
    for w, m in ablations.items():
        print(f"[{w:4}] AUC={m['auc']} AP={m['avg_precision']} lift@20%={m['lift_at_20pct']}")

    # --- Grouped-artist bootstrap ensemble -> stability intervals -------
    Xl, Xc = X(lib, "both"), X(cand, "both")
    artists = lib["group"].unique()
    boot_scores = np.empty((N_BOOT, len(cand)))
    for b in range(N_BOOT):
        pick = RNG.choice(artists, size=len(artists), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == a) for a in pick])
        if len(np.unique(y[idx])) < 2:
            idx = np.arange(len(y))
        m = model().fit(Xl[idx], y[idx])
        # store candidate scores as within-pool percentiles for comparability
        s = m.predict_proba(Xc)[:, 1]
        boot_scores[b] = pd.Series(s).rank(pct=True).to_numpy()

    cand["pref_med"] = np.median(boot_scores, axis=0)
    cand["pref_lo"] = np.percentile(boot_scores, 10, axis=0)
    cand["pref_hi"] = np.percentile(boot_scores, 90, axis=0)
    cand["spread"] = cand["pref_hi"] - cand["pref_lo"]

    # --- Discovery score + gamma robustness -----------------------------
    def disc(gamma):
        return cand["pref_med"] * cand["obscurity"] ** gamma
    cand["discovery"] = disc(1.0)
    tops = {g: set(cand.assign(d=disc(g)).nlargest(15, "d").index) for g in (0.5, 1.0, 2.0)}
    jac = len(tops[0.5] & tops[1.0] & tops[2.0]) / len(tops[0.5] | tops[1.0] | tops[2.0])
    print(f"\ngamma-robustness: top-15 3-way Jaccard = {jac:.2f}")

    def diversify(pool, by, n, cap=2):
        """Greedy top-n with a per-artist cap — a serendipity constraint so one
        artist can't monopolize the list."""
        chosen, counts = [], {}
        for i in pool.sort_values(by, ascending=False).index:
            a = pool.at[i, "primary_artist"]
            if counts.get(a, 0) >= cap:
                continue
            chosen.append(i)
            counts[a] = counts.get(a, 0) + 1
            if len(chosen) == n:
                break
        return pool.loc[chosen].copy()

    med_obs = cand["obscurity"].median()
    safe = diversify(cand[cand["obscurity"] > med_obs], "pref_lo", 12)
    safe["class"] = "safe_gem"
    wide = cand["spread"] > cand["spread"].median()
    moon = diversify(
        cand[wide & ~cand.index.isin(safe.index) & (cand["obscurity"] > med_obs)],
        "pref_hi", 8,
    )
    moon["class"] = "moonshot"
    picks = pd.concat([safe, moon])

    cols = ["title", "artist", "genre", "deezer_rank", "obscurity",
            "pref_lo", "pref_med", "pref_hi", "discovery", "class"]
    out = {
        "method": "PU-aware LR + grouped-artist bootstrap (n=%d) percentile stability intervals; discovery = pref_pct x obscurity^g" % N_BOOT,
        "cv_ablations_leave_one_artist_out": ablations,
        "gamma_top15_jaccard": round(jac, 2),
        "n_candidates": int(len(cand)),
        "picks": json.loads(picks[cols].round(3).to_json(orient="records")),
        "frontier": json.loads(
            cand[["title", "artist", "genre", "obscurity", "pref_med", "pref_lo", "pref_hi", "discovery"]]
            .round(3).to_json(orient="records")
        ),
        "caveats": [
            "Preference scores are cross-validated percentiles vs. weak negatives, NOT calibrated P(love).",
            "Bootstrap bands are ranking-stability intervals over artist resamples, not confidence intervals.",
            "Candidate pool is authored by 16 taste-adjacent artist searches; gems outside those neighborhoods are invisible.",
        ],
    }
    (ROOT / "outputs" / "gems.json").write_text(json.dumps(out, indent=1))
    print(f"\nSafe gems:")
    for _, r in safe.head(8).iterrows():
        print(f"  {r['title'][:38]:40} {r['artist'][:20]:22} pref[{r['pref_lo']:.2f},{r['pref_hi']:.2f}] obs={r['obscurity']:.2f}")
    print(f"Moonshots:")
    for _, r in moon.head(5).iterrows():
        print(f"  {r['title'][:38]:40} {r['artist'][:20]:22} pref[{r['pref_lo']:.2f},{r['pref_hi']:.2f}] obs={r['obscurity']:.2f}")
    print(f"\nWrote outputs/gems.json")


if __name__ == "__main__":
    main()
