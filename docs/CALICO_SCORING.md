### Calico — Scoring Rules (Detailed)

This document describes only the scoring-related rules for Calico. It explains each scoring concept, how and when points are awarded, and enumerates all pre-game parameters that affect scoring beyond the final placement of patch tiles.

References for rules language and terms: publisher overview and widely cited rules summaries (e.g., AEG/Flatout Games materials and BoardGameGeek rules references).

---

### Core Concepts and Definitions

- **Patch tile**: A hex with exactly one color and one pattern.
- **Adjacency / contiguous**: Two hexes are adjacent if they share an edge. A contiguous group is a set of tiles connected by edge adjacency.
- **Color group**: A contiguous group of tiles all sharing the same color (used for buttons). There are 6 different colors in total.
- **Pattern group**: A contiguous group of tiles all sharing the same pattern (used for cats). There are 6 different patterns in total.
- **Design goal neighborhood**: The six hexes that surround one design goal tile on a player’s board. These six specific positions are used to evaluate that goal’s arrangement.
- **Letter notation (design goals)**: Letters indicate “same vs different” among the six surrounding tiles. For example, `AA-BB-CC` means three pairs where each pair matches (by color or by pattern), and pairs are different from one another; `A-B-C-D-E-F` means all six different; `AAA-BBB` means two triplets of matching tiles, and the two triplets are different from each other.
- **Game board**: each game board (or Quilt Board) is a rectangle board consisting of hexagonal tiles putting together without gaps (hence each tile is neighboured to 6 other tiles). The board is 7x7, with 24 tiles are edges and 25 inner tiles. Among the inner tiles, 3 are design goal tiles, where the scoring rules are placed, leaving 22 left to place the patch tiles. These 3 tiles are in (col-row) positions: (4,3), (5,4), and (3,5).

---

### Scoring Categories

Calico scores points exclusively from three categories:

1) Buttons (color groups)
2) Cats (pattern groups meeting a cat’s requirement)
3) Design goal tiles (arrangements around each player’s 3 goal tiles)

At game end, total the points from these three categories. Tie-break rules are listed at the end.

---

### 1) Buttons (Color-Based Scoring)

- **Trigger**: Immediately when a placement creates a contiguous color group of size ≥ 3, take a button of that color and place it on one tile in that group.
- **Distinct groups vs growth**:
  - You receive **one** button per **distinct** contiguous color group upon first reaching size ≥ 3.
  - If you later expand that exact same contiguous group (e.g., 3 → 4 → 5 tiles), you do **not** receive additional buttons for that group.
  - If you create a **separate** contiguous group (not connected to the first) of the same color and it reaches size ≥ 3, you may gain **another** button of that same color.
- **Rainbow button**: If you have collected one button of **each** of the six colors, gain a single rainbow button bonus.
- **Scoring values**:
  - Each color button = 3 points
  - Rainbow button = +3 points (one-time bonus)

Notes:
- Color groups are determined solely by color, independent of patterns.
- A patch tile may contribute to both a color group (button) and a pattern group (cat) simultaneously because those track different attributes.

---

### 2) Cats (Pattern-Based Scoring)

- **What defines a qualifying group**:
  - A contiguous group of tiles all sharing the **same pattern**.
  - The group must meet the **requirement** shown on the active cat scoring tile (e.g., minimum size N; or a specific shape/formation), and it must use **one** of the two allowed patterns assigned to that cat during setup.
- **Claiming a cat token**:
  - When a qualifying group is created by your placement, take the **highest-value remaining** token for that cat and place it on a tile in the qualifying group.
  - You may claim the same cat **multiple times** if you later create another **distinct** qualifying group for that cat.
  - Cat tokens for a given cat are limited by its token stack; values typically decrease as tokens are taken.
- **Overlaps and independence**:
  - Pattern groups are defined independently of color groups. It’s common for the same placed tile to contribute to a color group (for a button) and a pattern group (for a cat) at the same time.
  - Whether two pattern groups are considered distinct is based on contiguity: distinct groups are not edge-connected to one another.
- **Scoring values**:
  - Each cat token is worth the printed value on that token (values differ by cat tile and diminish as more are taken).

---

### 3) Design Goal Tiles (Arrangement Scoring)

Each player has exactly **three** design goal tiles on their board. Each goal evaluates the **six surrounding hexes** once those six are placed.

- **How to satisfy a goal**:
  - Each goal shows a letter pattern for the six adjacent positions (e.g., `AA-BB-CC`, `AAA-BBB`, `A-B-C-D-E-F`, etc.).
  - You can satisfy this requirement **by color** or **by pattern**:
    - Satisfy by **color only** → score the lower value shown on the goal tile.
    - Satisfy by **pattern only** → score the lower value.
    - Satisfy **both** color and pattern simultaneously on the same six hexes → score the **higher** value.
  - If neither color nor pattern satisfies the requirement, score 0 for that goal.
- **Timing**:
  - A goal is evaluated as soon as all six surrounding spaces are filled. Tiles never move in Calico, so once satisfied the score is effectively fixed.
- **Independence**:
  - Design goals are independent of button and cat scoring. The same six tiles can simultaneously contribute to button groups and/or cat groups; that does not affect the goal’s evaluation except insofar as the tiles’ colors/patterns must match the goal’s letter notation.
- **Scoring values**:
  - Each goal tile has two numbers printed on it: a lower value (color **or** pattern) and a higher value (color **and** pattern). Concrete values vary by which three goal tiles were chosen during setup.

---

### End-of-Game Tally and Tie-Breakers

At game end (all boards filled), each player’s score is:

- Sum of points from all earned **buttons** (3 per color button + 3 for rainbow if achieved)
- Sum of points from all **cat tokens** taken (per-token printed values)
- Sum of points from the three **design goal tiles** (lower or higher value per tile, or 0 if not achieved)

Tie-breakers (if needed):
- Most buttons → then most cats → then share the win (if still tied).

---

### Pre-Game Parameters That Affect Scoring

When computing or validating scores, record these parameters (they are decided before or during setup and determine how points can be earned):

1) Global/Central (apply to all players)
   - **Active cat tiles (3 total)**:
     - For each cat tile in play:
       - Cat tile identity (name/id) and chosen **side** (each side may specify different requirements/values).
       - The cat’s **requirement**: minimum size and/or shape/formation used for qualifying groups.
       - The two **allowed patterns** (selected/assigned using the black & white pattern tiles during setup).
       - The cat’s **token value stack** (an ordered list of remaining token values, descending). This is the supply you draw from when a player qualifies.
   - **Button colors**:
     - The six distinct colors in play (standard set). Each color has an available supply of buttons; buttons score a flat 3 points each. The **rainbow button** bonus is enabled when a player possesses all six different color buttons.

2) Per Player
   - **Design goals (3 tiles)**:
     - For each of the player’s three goal tiles:
       - Goal tile identity (name/id).
       - The **letter arrangement** requirement among the six surrounding positions (e.g., `AA-BB-CC`, `AAA-BBB`, `A-B-C-D-E-F`, etc.).
       - The **lower** and **higher** point values printed on that goal tile (lower = color or pattern; higher = color and pattern).
       - The physical **location** of each goal tile on the board (to identify which six neighbor hexes are evaluated for that goal).

3) Variants / Mode Flags (if applicable)
   - **Family mode**: If enabled, ignore all design goal scoring (only cats and buttons are scored).
   - **Solo mode**: Market/discard behavior differs, but scoring categories and values remain the same. Still record the same cat tiles and design goals because they determine scoring.
   - Any table-agreed **house rules** that alter button values, rainbow bonus, cat token stacks, or goal scoring should be captured explicitly.

---

### Practical Scoring Checklist (for a solver or score sheet)

Capture before play:
- [ ] Selected 3 cat tiles (with side) and their two allowed patterns each
- [ ] For each selected cat: the ordered token value stack available
- [ ] Each player’s 3 design goal tiles with: id, letter requirement, lower and higher point values, and board positions
- [ ] The six button colors in use (standard) and whether rainbow bonus is enabled (standard = yes)
- [ ] Mode flags: Family (goals off) / Solo (goals on; market changes only)

Compute at end of game (given final board states):
- Buttons: Identify all distinct color groups with size ≥ 3 that triggered during play; total 3 points per distinct group-button, plus 3 if all six colors earned (rainbow)
- Cats: For each active cat, find all distinct qualifying pattern groups; assign highest remaining token values per claim; sum token values
- Design goals: For each of a player’s 3 goals, evaluate the six neighbors by color and by pattern vs the letter requirement; award lower value if one attribute matches, higher if both, else 0

---

### Notes and Clarifications

- A single tile can simultaneously contribute to a color group (button) and a pattern group (cat) because those categories are evaluated on different attributes.
- Distinctness for groups is about **contiguity**: two groups of the same color (or same pattern) are distinct only if they are not edge-connected.
- Cat token values are limited by supply and typically decrease for subsequent claims. Always take the highest-value remaining token for that cat when you qualify.
- Design goals are evaluated on the specific six neighboring hexes around each goal tile. Scoring a goal does not reduce or change button/cat eligibility.

---

### Final Score Formula (per player)

Total Score = (3 × number_of_color_buttons) + (3 if rainbow_button_earned else 0) + (sum of all cat token values taken) + (sum of the three design goal scores)
