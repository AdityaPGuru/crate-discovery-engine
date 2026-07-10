"""Enrich Spotify track metadata via the free iTunes Search API.

The Spotify audio-features endpoint is deprecated, so we recover what we can
from public metadata instead: artist (missing for lazy-loaded tracks), genre,
release year, and album. Results are cached so re-runs are free.
"""

import difflib
import json
import re
import time
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_tracks.json"
CACHE = ROOT / "data" / "enrichment_cache.json"
OUT = ROOT / "data" / "tracks_enriched.json"

ITUNES_URL = "https://itunes.apple.com/search"
THROTTLE_S = 0.6

FEAT_RE = re.compile(r"\s*[\(\[](?:feat|ft|with|from)\.?[^\)\]]*[\)\]]", re.I)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = FEAT_RE.sub("", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("_", " "))
    return re.sub(r"\s+", " ", s).strip()


def load_tracks() -> dict:
    raw = json.loads(RAW.read_text())
    tracks: dict[str, dict] = {}
    for source, payload in raw["sources"].items():
        for t in payload["tracks"]:
            uri = t["uri"]
            rec = tracks.setdefault(
                uri,
                {
                    "uri": uri,
                    "title": t["title"],
                    "artist": None,
                    "duration_ms": None,
                    "explicit": None,
                    "is_saved": False,
                    "sources": [],
                },
            )
            rec["sources"].append(source)
            for field in ("artist", "duration_ms", "explicit"):
                if rec[field] is None and t.get(field) is not None:
                    rec[field] = t[field]
            if t.get("is_saved"):
                rec["is_saved"] = True
    return tracks


def search_itunes(title: str, artist: str | None) -> list[dict]:
    term = f"{title} {artist}" if artist else title
    resp = requests.get(
        ITUNES_URL,
        params={"term": term, "entity": "song", "limit": 5, "country": "US"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def best_match(title: str, artist: str | None, results: list[dict]) -> tuple[dict | None, float]:
    """Pick the result whose title (and artist, when known) best matches."""
    best, best_score = None, 0.0
    for r in results:
        score = difflib.SequenceMatcher(
            None, normalize(title), normalize(r.get("trackName", ""))
        ).ratio()
        if artist:
            a_score = difflib.SequenceMatcher(
                None, normalize(artist.split(",")[0]), normalize(r.get("artistName", ""))
            ).ratio()
            score = 0.65 * score + 0.35 * a_score
        if score > best_score:
            best, best_score = r, score
    return best, best_score


def main() -> None:
    tracks = load_tracks()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    for i, (uri, rec) in enumerate(tracks.items()):
        if uri in cache:
            continue
        try:
            results = search_itunes(rec["title"], rec["artist"])
            match, conf = best_match(rec["title"], rec["artist"], results)
        except requests.RequestException as e:
            print(f"  ! {rec['title']}: {e}")
            match, conf = None, 0.0
        cache[uri] = (
            {
                "artistName": match.get("artistName"),
                "genre": match.get("primaryGenreName"),
                "releaseDate": match.get("releaseDate"),
                "trackTimeMillis": match.get("trackTimeMillis"),
                "collectionName": match.get("collectionName"),
                "match_confidence": round(conf, 3),
            }
            if match and conf >= 0.5
            else {"match_confidence": round(conf, 3)}
        )
        CACHE.write_text(json.dumps(cache, indent=1))
        print(f"[{i + 1}/{len(tracks)}] {rec['title']} -> conf {conf:.2f}")
        time.sleep(THROTTLE_S)

    matched = 0
    for uri, rec in tracks.items():
        info = cache.get(uri, {})
        rec["match_confidence"] = info.get("match_confidence", 0.0)
        if info.get("genre"):
            matched += 1
        rec["genre"] = info.get("genre")
        rec["album"] = info.get("collectionName")
        year = info.get("releaseDate")
        rec["release_year"] = int(year[:4]) if year else None
        if rec["artist"] is None and info.get("artistName"):
            rec["artist"] = info["artistName"]
        if rec["duration_ms"] is None and info.get("trackTimeMillis"):
            rec["duration_ms"] = info["trackTimeMillis"]

    OUT.write_text(json.dumps(list(tracks.values()), indent=1))
    print(f"\nEnriched {matched}/{len(tracks)} tracks with genre ({matched / len(tracks):.0%}).")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
