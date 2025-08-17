### Calico — Design Goal Tiles Reference

This file enumerates all design goal tile configurations defined for Calico. There are 24 total design goal tiles in the box, consisting of 4 identical sets of 6 unique configurations (one set per player). During setup, each player shuffles their personal 6 goal tiles, draws 4, and chooses 3 to use.

Letter notation primer:
- Letters indicate “same vs different” across the six hexes surrounding a goal tile.
- Example: `AA-BB-CC` means three pairs; each pair matches internally (A with A, B with B, C with C), and the three pairs are mutually different from one another.
- A configuration may be satisfied by COLOR or by PATTERN (or by both simultaneously); see `docs/CALICO_SCORING.md` for scoring specifics.

---

### The 6 Unique Configurations

1) `A-B-C-D-E-F` (Six Unique)
   - All six surrounding hexes are mutually different.
   - Example by color: six different colors; by pattern: six different patterns.
   - Scoring printed on tile: 10 (color-or-pattern) / 15 (color-and-pattern)

2) `AA-BB-CC` (Three Pairs)
   - Three distinct pairs among the six surrounding hexes.
   - Example by color: two blues, two yellows, two purples; by pattern: two polka, two stripe, two floral.
    - Scoring printed on tile: 7 (color-or-pattern) / 11 (color-and-pattern)

3) `AAA-BBB` (Two Triplets)
   - Two distinct sets of three matching hexes.
   - Example by color: three teal + three pink; by pattern: three herringbone + three chevron.
    - Scoring printed on tile: 8 (color-or-pattern) / 13 (color-and-pattern)

4) `AAA-BB-C` (3–2–1)
   - One triplet, one pair, and one single (all three groups mutually different).
   - Example by color: three green + two orange + one purple; by pattern: three dots + two plaid + one wave.
    - Scoring printed on tile: 7 (color-or-pattern) / 11 (color-and-pattern)

5) `AA-BB-C-D` (2–2–1–1)
   - Two distinct pairs plus two singles (all four groups mutually different).
   - Example by color: two red + two blue + one yellow + one teal; by pattern: two leaf + two check + one stitch + one swirl.
    - Scoring printed on tile: 5 (color-or-pattern) / 8 (color-and-pattern)

6) `AAAA-BB` (4–2)
   - One set of four and one pair (both groups mutually different).
   - Example by color: four purple + two orange; by pattern: four quilt + two diamond.
    - Scoring printed on tile: 8 (color-or-pattern) / 14 (color-and-pattern)

---

### Counts and Duplication

- Total tiles in box: 24
- Unique configurations: 6 (listed above)
- Copies per configuration: 4 (one copy in each player’s personal set of 6)

---

### Notes

- Each player’s personal set contains exactly one of each of the six configurations above.
- The physical orientation of surrounding hexes does not change the configuration; only the multiset structure (e.g., 3–2–1) matters.
- Scoring values (lower for color-or-pattern; higher for color-and-pattern) are printed on each goal tile and depend on the configuration; see `docs/CALICO_SCORING.md` for how goals are evaluated and scored.


