# You Will Love These — model-ranked picks

Leave-artist-out CV: LR AUC **0.615**, GBT AUC **0.561**

| # | Track | Artist | Genre | Year | Score |
|---|-------|--------|-------|------|-------|
| 1 | Feel Good Inc. | Gorillaz | Alternative | 2005 | 0.784 |
| 2 | Nikes on My Feet | Mac Miller | Hip-Hop/Rap | 2010 | 0.726 |
| 3 | Tongue Tied | Grouplove | Alternative | 2011 | 0.652 |
| 4 | A COLD PLAY | nan | ? | ? | 0.639 |
| 5 | I Wanna Be Yours | Arctic Monkeys | Alternative | 2013 | 0.597 |
| 6 | Drugs You Should Try It | Travis Scott | Hip-Hop/Rap | 2014 | 0.582 |
| 7 | HIGHS AND LOWS | nan | ? | ? | 0.567 |
| 8 | Riptide | Vance Joy | Alternative | 2013 | 0.565 |
| 9 | LIL DEMON | nan | ? | ? | 0.561 |
| 10 | 大展鴻圖(Blueprint Supreme) | nan | ? | ? | 0.556 |
| 11 | KING | nan | ? | ? | 0.549 |
| 12 | Come a Little Closer | Cage The Elephant | Alternative | 2013 | 0.548 |
| 13 | BOY IN RED | nan | ? | ? | 0.534 |
| 14 | Eternity | nan | ? | ? | 0.53 |
| 15 | HA | nan | ? | ? | 0.523 |
| 16 | Potential | nan | ? | ? | 0.522 |
| 17 | ALL THE LOVE (feat. Andre Troutman) | nan | ? | ? | 0.519 |
| 18 | keep steady | nan | ? | ? | 0.517 |
| 19 | Way Too Self Aware | nan | ? | ? | 0.516 |
| 20 | Victory Lap | nan | ? | ? | 0.51 |

## Caveats
- PU learning: 'negatives' are unsaved tracks from Spotify's own personalized mixes — plausible likes, not true dislikes. Reported AUC is a lower bound.
- Tiny dataset (~170 tracks); metrics have wide confidence intervals.
- Candidate pool is pre-filtered by Spotify's recommender (popularity/selection bias).