"""Task 3: Heuristic Pathfinding Agent (Dijkstra + A* implemented from scratch)
Grid: 0 = free, 1 = obstacle. Moves: 4-directional, cost 1 per step."""
import heapq, time, random, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# ---------- Heuristics ----------
def manhattan(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
def euclidean(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
def zero(a, b): return 0   # h = 0 turns A* into Dijkstra

# ---------- Core search (priority queue) ----------
def search(grid, start, goal, heuristic):
    """Generic best-first search. f = g + h. Returns path, expanded-order list, runtime(ms)."""
    t0 = time.perf_counter()
    rows, cols = grid.shape
    g = {start: 0}
    parent = {start: None}
    heap = [(heuristic(start, goal), 0, start)]
    closed, order = set(), []
    while heap:
        f, cost, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur); order.append(cur)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur); cur = parent[cur]
            return path[::-1], order, (time.perf_counter() - t0) * 1000
        for dr, dc in MOVES:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and grid[nxt] == 0:
                ng = g[cur] + 1
                if ng < g.get(nxt, float("inf")):
                    g[nxt] = ng; parent[nxt] = cur
                    heapq.heappush(heap, (ng + heuristic(nxt, goal), ng, nxt))
    return None, order, (time.perf_counter() - t0) * 1000

def dijkstra(grid, s, g): return search(grid, s, g, zero)
def astar_manhattan(grid, s, g): return search(grid, s, g, manhattan)
def astar_euclid(grid, s, g): return search(grid, s, g, euclidean)

ALGOS = {"Dijkstra": dijkstra, "A* (Manhattan)": astar_manhattan, "A* (Euclidean)": astar_euclid}

# ---------- Grid generators ----------
def random_grid(n, density, seed):
    """Random obstacles; the seed is bumped until a path from corner to corner exists."""
    while True:
        rng = np.random.default_rng(seed)
        grid = (rng.random((n, n)) < density).astype(int)
        grid[0, 0] = grid[n - 1, n - 1] = 0
        if search(grid, (0, 0), (n - 1, n - 1), zero)[0] is not None:
            return grid
        seed += 100

def maze_grid(n, seed):
    """Recursive-backtracker maze on an odd-sized grid (guaranteed solvable)."""
    random.seed(seed)
    n = n if n % 2 else n + 1
    grid = np.ones((n, n), dtype=int)
    stack = [(0, 0)]; grid[0, 0] = 0
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr * 2, c + dc * 2, dr, dc) for dr, dc in MOVES
                if 0 <= r + dr * 2 < n and 0 <= c + dc * 2 < n and grid[r + dr * 2, c + dc * 2] == 1]
        if nbrs:
            nr, nc, dr, dc = random.choice(nbrs)
            grid[r + dr, c + dc] = 0; grid[nr, nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()
    return grid

def wall_with_gap(n):
    grid = np.zeros((n, n), dtype=int)
    grid[n // 2, :] = 1; grid[n // 2, n // 4] = 0   # wall with one small gap
    grid[:n // 2, n // 2] = 1; grid[n // 4, n // 2] = 0
    return grid

SCENARIOS = {
    "Open grid (10% obstacles)": random_grid(30, 0.10, 1),
    "Medium (25% obstacles)": random_grid(30, 0.25, 2),
    "Dense (35% obstacles)": random_grid(30, 0.35, 3),
    "Wall with gap": wall_with_gap(30),
    "Maze": maze_grid(29, 4),
}

# ---------- Visualization ----------
CMAP = ListedColormap(["white", "#333333", "#cfe2f3"])  # free, wall, explored

def draw(ax, grid, order, path, title):
    show = grid.copy()
    for r, c in order: show[r, c] = 2
    ax.imshow(show, cmap=CMAP, vmin=0, vmax=2)
    if path:
        ys, xs = zip(*path); ax.plot(xs, ys, color="red", lw=2)
    ax.plot(0, 0, "go", ms=7); ax.plot(grid.shape[1] - 1, grid.shape[0] - 1, "b*", ms=10)
    ax.set_title(title, fontsize=8); ax.axis("off")

def main():
    log = []
    for name, grid in SCENARIOS.items():
        n = grid.shape[0]
        start, goal = (0, 0), (n - 1, n - 1)
        free = int((grid == 0).sum())
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        for ax, (aname, fn) in zip(axes, ALGOS.items()):
            # average runtime over 20 runs for stable timing
            runs = [fn(grid, start, goal) for _ in range(20)]
            path, order, _ = runs[0]
            ms = float(np.mean([r[2] for r in runs]))
            steps = len(path) - 1 if path else None
            density = len(order) / free
            log.append({"scenario": name, "algorithm": aname, "solved": path is not None,
                        "steps": steps, "expanded_nodes": len(order),
                        "expanded_density_%": round(density * 100, 1), "runtime_ms": round(ms, 3)})
            draw(ax, grid, order, path,
                 f"{aname}\nsteps={steps} expanded={len(order)} ({density:.0%}) {ms:.2f} ms")
        fig.suptitle(name); plt.tight_layout()
        fname = name.split("(")[0].strip().lower().replace(" ", "_")
        plt.savefig(f"outputs/viz_{fname}.png", dpi=130); plt.close()

    with open("outputs/metrics_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys()); w.writeheader(); w.writerows(log)

    # summary chart: expanded nodes per algorithm
    scen = list(SCENARIOS); algs = list(ALGOS); x = np.arange(len(scen))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, a in enumerate(algs):
        ex = [next(r["expanded_nodes"] for r in log if r["scenario"] == s and r["algorithm"] == a) for s in scen]
        rt = [next(r["runtime_ms"] for r in log if r["scenario"] == s and r["algorithm"] == a) for s in scen]
        axes[0].bar(x + i * 0.27, ex, 0.27, label=a); axes[1].bar(x + i * 0.27, rt, 0.27, label=a)
    for ax, t in zip(axes, ["Expanded nodes", "Runtime (ms)"]):
        ax.set_xticks(x + 0.27); ax.set_xticklabels([s.split("(")[0].strip() for s in scen], rotation=20, fontsize=8)
        ax.set_title(t); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("outputs/summary_chart.png", dpi=130); plt.close()

    # GIF: A* exploring the maze step by step
    grid = SCENARIOS["Maze"]; n = grid.shape[0]
    path, order, _ = astar_manhattan(grid, (0, 0), (n - 1, n - 1))
    fig, ax = plt.subplots(figsize=(4, 4))
    frames = list(range(0, len(order), max(1, len(order) // 60))) + [len(order)]
    def update(k):
        ax.clear(); draw(ax, grid, order[:k], path if k >= len(order) else None, f"A* exploring: {k} nodes")
    FuncAnimation(fig, update, frames=frames).save("outputs/astar_maze.gif", writer=PillowWriter(fps=8))
    plt.close()

    print(f"{'Scenario':28s}{'Algorithm':17s}{'Steps':>6s}{'Expanded':>9s}{'Dens%':>7s}{'ms':>8s}")
    for r in log:
        print(f"{r['scenario']:28s}{r['algorithm']:17s}{str(r['steps']):>6s}{r['expanded_nodes']:>9d}"
              f"{r['expanded_density_%']:>7}{r['runtime_ms']:>8}")

if __name__ == "__main__":
    main()
