# Task 3: Heuristic Pathfinding Agent

Dijkstra and A* search written **from scratch** (no pathfinding libraries) to navigate grid mazes from the top-left corner to the bottom-right corner.

## Algorithms
| Algorithm | Idea |
|---|---|
| Dijkstra | Expands nodes by lowest cost so far g(n). Guarantees shortest path, no direction guidance. |
| A* (Manhattan) | Uses f(n) = g(n) + h(n) with h = Manhattan distance. Admissible for 4-direction moves. |
| A* (Euclidean) | Same, with straight-line distance (admissible but weaker for grid moves). |

All three share one priority-queue search (`search()`); only the heuristic changes (h = 0 gives Dijkstra).

## Test Scenarios (30x30 grids)
Open grid (10% obstacles), Medium (25%), Dense (35%), Wall with gap, and a generated Maze.

## Metrics tracked
Path length (steps), runtime (ms, averaged over 20 runs), expanded nodes, expanded-node density (% of free cells explored).

## Structure
```
task3_pathfinding/
├── pathfinding.py        # algorithms + experiments + visualization
├── outputs/
│   ├── metrics_log.csv   # all metrics
│   ├── viz_*.png         # side-by-side path/explored-area plots per scenario
│   ├── summary_chart.png # expanded nodes + runtime comparison
│   └── astar_maze.gif    # animation of A* exploring the maze
├── requirements.txt
└── README.md
```

## Run
```bash
pip install -r requirements.txt
python pathfinding.py
```

## Key Results
- All algorithms found the same shortest path length in every scenario (A* with an admissible heuristic is optimal).
- A* (Manhattan) expanded 40% of free cells vs 99% for Dijkstra on the 25% obstacle grid, and about 2.2x fewer nodes on the dense grid.
- In the maze (one narrow route) the benefit is small: 235 vs 261 expanded nodes.
- Q-Learning was not included; the task allows A*, Dijkstra, or Q-Learning.
