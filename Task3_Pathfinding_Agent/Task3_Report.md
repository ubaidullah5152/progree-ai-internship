# Task 3 Report: Heuristic Graph Pathfinding Agent

**Intern:** Ubaid Ullah  |  **Organization:** Progree  |  **Domain:** Artificial Intelligence

## 1. Objective
Implement heuristic pathfinding algorithms from scratch and compare their efficiency across grids with different obstacle layouts.

## 2. Method
- **Environment:** 30x30 grids (0 = free, 1 = obstacle), 4-direction moves, cost 1 per move. Start (0,0), goal (bottom-right).
- **Algorithms:** Dijkstra, A* with Manhattan heuristic, A* with Euclidean heuristic, all using a binary-heap priority queue (`heapq`) and a closed set.
- **Scenarios:** 10%, 25% and 35% random obstacles (seeds adjusted so a path exists), a wall with a gap, and a recursive-backtracker maze.
- **Metrics:** steps in the path, expanded nodes, expanded density (expanded / free cells), runtime (mean of 20 runs).

## 3. Results
| Scenario | Algorithm | Steps | Expanded | Density | Runtime (ms) |
|---|---|---|---|---|---|
| Open 10% | Dijkstra | 58 | 815 | 100% | 3.02 |
| Open 10% | A* Manhattan | 58 | 702 | 86% | 3.14 |
| Open 10% | A* Euclidean | 58 | 787 | 97% | 3.26 |
| Medium 25% | Dijkstra | 58 | 659 | 99% | 2.21 |
| Medium 25% | A* Manhattan | 58 | 268 | 40% | 1.18 |
| Medium 25% | A* Euclidean | 58 | 470 | 71% | 1.93 |
| Dense 35% | Dijkstra | 58 | 538 | 89% | 1.88 |
| Dense 35% | A* Manhattan | 58 | 239 | 40% | 0.88 |
| Dense 35% | A* Euclidean | 58 | 383 | 63% | 1.55 |
| Wall with gap | Dijkstra | 58 | 857 | 100% | 2.74 |
| Wall with gap | A* Manhattan | 58 | 661 | 77% | 1.90 |
| Wall with gap | A* Euclidean | 58 | 725 | 85% | 2.56 |
| Maze | Dijkstra | 204 | 261 | 58% | 0.46 |
| Maze | A* Manhattan | 204 | 235 | 52% | 0.43 |
| Maze | A* Euclidean | 204 | 235 | 52% | 0.47 |

Figures: `outputs/viz_*.png`, `outputs/summary_chart.png`, `outputs/astar_maze.gif`; raw data in `outputs/metrics_log.csv`.

## 4. Analysis
1. **Optimality:** every algorithm returned the same path length per scenario, confirming both heuristics are admissible.
2. **Efficiency:** A* with Manhattan distance explored the fewest nodes in most scenarios (about 40% of free cells in the medium and dense grids vs ~90-99% for Dijkstra), and ran about 1.5-2.5x faster there.
3. **Heuristic quality:** Manhattan is closer to the true cost on 4-direction grids than Euclidean, so it prunes more. Euclidean sits between Manhattan and Dijkstra.
4. **Open grid:** with few obstacles the heuristic helps less (many equally good paths), and runtime was about equal; the small differences are within timing noise because the heap and heuristic calls add overhead.
5. **Maze:** a maze has essentially one route, so a heuristic cannot guide much; all algorithms explored a similar number of nodes.

## 5. Limitations
Single start/goal per scenario, one random grid per obstacle density, and millisecond timings that vary by machine. Averaging over many random grids would give stronger statistics.

## 6. Future Work
Add Q-Learning, 8-direction movement with diagonal costs, weighted terrain, and larger grids.

## 7. Conclusion
A* with an admissible, well-matched heuristic finds optimal paths while expanding far fewer nodes than Dijkstra when the grid is not a single-corridor maze.
