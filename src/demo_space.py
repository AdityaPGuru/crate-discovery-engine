"""Build the client-side demo space for the 'build your crate' feature.

Any visitor picks seed songs they love; the page computes cosine similarity
between the seed centroid and every candidate, entirely in the browser.
This script precomputes the shared vector space: TF-IDF (char ngrams over
title+artist) + scaled metadata, reduced to 16 dims with TruncatedSVD, for
both the 321 Deezer candidates and 24 widely-known seed songs.

Honest framing: the demo is a similarity engine in taste-space, not the
per-user PU model (which needs a listening history to train on).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from gem_hunter import GENRES, load_candidates, load_library

ROOT = Path(__file__).resolve().parent.parent

# (title, artist, genre, year, duration_s, explicit) — famous, cluster-spanning.
SEEDS = [
    ("Viva La Vida", "Coldplay", "Alternative", 2008, 242, False),
    ("Mr. Brightside", "The Killers", "Alternative", 2004, 222, False),
    ("Electric Feel", "MGMT", "Alternative", 2007, 229, False),
    ("Riptide", "Vance Joy", "Alternative", 2013, 204, False),
    ("Sweater Weather", "The Neighbourhood", "Alternative", 2012, 240, False),
    ("Heat Waves", "Glass Animals", "Alternative", 2020, 238, False),
    ("505", "Arctic Monkeys", "Alternative", 2007, 253, False),
    ("I Wanna Be Yours", "Arctic Monkeys", "Alternative", 2013, 184, False),
    ("Blinding Lights", "The Weeknd", "Pop", 2019, 200, False),
    ("As It Was", "Harry Styles", "Pop", 2022, 167, False),
    ("Levitating", "Dua Lipa", "Pop", 2020, 203, False),
    ("Sunflower", "Post Malone, Swae Lee", "Pop", 2018, 158, False),
    ("SICKO MODE", "Travis Scott", "Hip-Hop/Rap", 2018, 312, True),
    ("HUMBLE.", "Kendrick Lamar", "Hip-Hop/Rap", 2017, 177, True),
    ("God's Plan", "Drake", "Hip-Hop/Rap", 2018, 199, True),
    ("Self Care", "Mac Miller", "Hip-Hop/Rap", 2018, 337, True),
    ("XO Tour Llif3", "Lil Uzi Vert", "Hip-Hop/Rap", 2017, 183, True),
    ("Falling Down", "Lil Peep, XXXTENTACION", "Hip-Hop/Rap", 2018, 196, True),
    ("Nights", "Frank Ocean", "R&B/Soul", 2016, 307, True),
    ("Kill Bill", "SZA", "R&B/Soul", 2022, 154, False),
    ("Get You", "Daniel Caesar, Kali Uchis", "R&B/Soul", 2016, 278, False),
    ("Best Part", "Daniel Caesar, H.E.R.", "R&B/Soul", 2017, 210, False),
    ("Dark Red", "Steve Lacy", "R&B/Soul", 2017, 173, False),
    ("Die For You", "The Weeknd", "R&B/Soul", 2016, 260, False),
]


def main() -> None:
    # Must mirror gem_hunter.main exactly so vec order == frontier order.
    cand = load_candidates()
    known = set(load_library()["title"].str.lower().str.strip())
    cand = cand[~cand["title"].str.lower().str.strip().isin(known)].reset_index(drop=True)
    seeds = pd.DataFrame(SEEDS, columns=["title", "artist", "genre", "year", "duration_s", "explicit"])
    both = pd.concat(
        [
            cand[["title", "artist", "genre", "year", "duration_s", "explicit"]],
            seeds[["title", "artist", "genre", "year", "duration_s", "explicit"]],
        ],
        ignore_index=True,
    )

    text = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=1200).fit_transform(
        (both["title"] + " " + both["artist"]).str.lower()
    )
    genre = pd.DataFrame({f"g_{k}": (both["genre"] == k).astype(float) for k in GENRES})
    meta = StandardScaler().fit_transform(
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "duration_s": both["duration_s"].astype(float),
                        "year": both["year"].fillna(both["year"].median()).astype(float),
                        "explicit": both["explicit"].astype(float),
                    }
                ),
                genre,
            ],
            axis=1,
        )
    )
    X = hstack([meta * 0.6, text]).tocsr()  # keep text dominant but metadata present
    Z = TruncatedSVD(n_components=16, random_state=0).fit_transform(X)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)  # unit vectors -> dot = cosine

    n = len(cand)
    out = {
        "dims": 16,
        "cand_vecs": np.round(Z[:n], 3).tolist(),
        "seeds": [
            {"title": t, "artist": a, "genre": g, "vec": np.round(Z[n + i], 3).tolist()}
            for i, (t, a, g, *_rest) in enumerate(SEEDS)
        ],
    }
    (ROOT / "outputs" / "demo_space.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"demo space: {n} candidate vecs + {len(SEEDS)} seeds, 16 dims")
    # sanity: nearest candidates to an indie seed should look indie
    zi = Z[n + 3]  # Riptide
    sim = Z[:n] @ zi
    top = np.argsort(-sim)[:5]
    for i in top:
        print("  Riptide ->", cand.iloc[i]["title"], "—", cand.iloc[i]["artist"], round(float(sim[i]), 2))


if __name__ == "__main__":
    main()
