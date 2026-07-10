"""Project the 16-dim taste space into NEBULA's 3D galaxy.

Mapping (honest, not decorative):
- Angular position: first 3 principal directions of the 16-dim embedding —
  songs that sound adjacent sit in the same region of sky.
- Radial distance from the galactic core: obscurity. The mainstream burns
  bright at the center; the unheard drift in the dark at the rim.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GENRE_IDX = {"Alternative": 0, "Hip-Hop/Rap": 1, "R&B/Soul": 2, "Pop": 3}


def main() -> None:
    demo = json.loads((ROOT / "outputs" / "demo_space.json").read_text())
    gems = json.loads((ROOT / "outputs" / "gems.json").read_text())
    V = np.array(demo["cand_vecs"])            # (321, 16), unit norm
    S = np.array([s["vec"] for s in demo["seeds"]])

    # PCA directions from the candidates; project both sets.
    _, _, Wt = np.linalg.svd(V - V.mean(0), full_matrices=False)
    P3 = Wt[:3].T
    def to_sky(M):
        d = M @ P3
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)

    cd, sd = to_sky(V), to_sky(S)
    stars = []
    for i, t in enumerate(gems["frontier"]):
        r = 0.30 + 0.70 * t["obscurity"]
        x, y, z = (cd[i] * r).round(3)
        stars.append([float(x), float(y), float(z), GENRE_IDX.get(t["genre"], 0),
                      round(t["obscurity"], 2), t["title"], t["artist"]])
    seeds = [
        {"title": s["title"], "artist": s["artist"],
         "xyz": [round(float(v), 3) for v in sd[i] * 0.22], "vec": s["vec"]}
        for i, s in enumerate(demo["seeds"])
    ]
    out = {"dims": demo["dims"], "stars": stars, "cand_vecs": demo["cand_vecs"], "seeds": seeds}
    (ROOT / "outputs" / "nebula_data.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"nebula: {len(stars)} stars, {len(seeds)} seed anchors")
    # cluster sanity: mean pairwise angle within vs across genres
    g = np.array([s[3] for s in stars])
    within = np.mean([cd[g == k] @ cd[g == k].T for k in range(3)][0])
    print("mean within-genre cos (alt):", round(float(within), 2))


if __name__ == "__main__":
    main()
