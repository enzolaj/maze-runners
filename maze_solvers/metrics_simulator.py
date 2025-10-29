import argparse
import csv
import os
import random
import statistics
import time

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from complete_file import Maze, Robot


ONLINE_ONLY = {"BFS", "Greedy BFS", "Depth-First Search", "Monte Carlo"}
OFFLINE_ONLY = {"Prim Solver", "Kruskal Solver"}

DEFAULT_ALGOS = [
    "A*",               # can be online (knows_maze=False)
    "Greedy BFS",
    "BFS",
    "Depth-First Search",
    "Monte Carlo",
    "Prim Solver",      # offline only
    "Kruskal Solver"    # offline only
]

DEFAULT_GENS = [
    "DFS",
    "BFS",
    "Greedy Frontier",
    "Random Prim's",
    "Weighted Prim's (MST)",
    "Kruskal's MST",
]


def run_single(width, height, gen_method, algorithm, loop_percent=0, num_rewards=0, max_weight=1, max_ticks=20000, seed=None):
    if seed is not None:
        random.seed(seed)

    # Determine knows_maze automatically
    knows_maze = False
    if algorithm in OFFLINE_ONLY:
        knows_maze = True
    elif algorithm in ONLINE_ONLY:
        knows_maze = False
    else:
        # A* can be either; default to online for fairness
        knows_maze = False

    maze = Maze(width, height, gen_method, loop_percent, num_rewards, max_weight)
    robot = Robot(maze, algorithm=algorithm, knows_maze=knows_maze)

    t0 = time.perf_counter()
    ticks = 0
    while not robot.is_done and ticks < max_ticks:
        robot.step()
        ticks += 1
    elapsed = time.perf_counter() - t0

    m = robot.metrics
    result = {
        "gen_method": gen_method,
        "algorithm": algorithm,
        "width": width,
        "height": height,
        "loop_percent": loop_percent,
        "num_rewards": num_rewards,
        "max_weight": max_weight,
        "knows_maze": knows_maze,
        "ticks": ticks,
        "elapsed_sec": elapsed,
        "steps": m.get("steps", 0),
        "algorithm_steps": m.get("algorithm_steps", 0),
        "unique_explored": m.get("unique_explored", 0),
        "nodes_expanded": m.get("nodes_expanded", 0),
        "path_length": m.get("path_length", 0),
        "total_cost": m.get("total_cost", 0),
        "frontier_max": m.get("frontier_max", 0),
        "finished": bool(robot.is_done),
    }
    return result


def aggregate_results(rows, group_by=("gen_method", "algorithm")):
    # Aggregate by group-by keys
    grouped = {}
    for r in rows:
        key = tuple(r[k] for k in group_by)
        grouped.setdefault(key, []).append(r)

    def agg_stat(values):
        if not values:
            return {"avg": 0, "min": 0, "max": 0, "stdev": 0}
        return {
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0,
        }

    metrics = [
        "elapsed_sec",
        "ticks",
        "steps",
        "algorithm_steps",
        "unique_explored",
        "nodes_expanded",
        "path_length",
        "total_cost",
        "frontier_max",
    ]

    summary = []
    for key, items in grouped.items():
        entry = {"group": key, "count": len(items)}
        for m in metrics:
            stats = agg_stat([it[m] for it in items])
            for stat_name, value in stats.items():
                entry[f"{m}_{stat_name}"] = value
        entry["finished_rate"] = sum(1 for it in items if it["finished"]) / len(items)
        summary.append(entry)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_metric(summary, metric_key, out_path):
    if not HAS_MPL:
        return
    # summary rows have key: (gen_method, algorithm)
    labels = []
    values = []
    for row in summary:
        if "group" in row:
            gen, algo = row["group"]
        else:
            gen, algo = row.get("gen_method", ""), row.get("algorithm", "")
        labels.append(f"{algo}\n({gen})")
        values.append(row.get(metric_key, 0))
    plt.figure(figsize=(max(8, len(labels)*0.6), 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel(metric_key)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run repeated maze simulations and plot metrics.")
    parser.add_argument("--gui", action="store_true", help="Open interactive GUI instead of headless run")
    parser.add_argument("--runs", type=int, default=10, help="Runs per (generator, algorithm) pair")
    parser.add_argument("--width", type=int, default=25)
    parser.add_argument("--height", type=int, default=25)
    parser.add_argument("--loop_percent", type=int, default=0)
    parser.add_argument("--num_rewards", type=int, default=0)
    parser.add_argument("--max_weight", type=int, default=1)
    parser.add_argument("--generators", nargs="*", default=DEFAULT_GENS)
    parser.add_argument("--algorithms", nargs="*", default=DEFAULT_ALGOS)
    parser.add_argument("--out_dir", default="metrics_output")
    args = parser.parse_args()

    if args.gui:
        # Launch the GUI app
        import metrics_app  # noqa: F401
        return

    all_rows = []
    seed_base = int(time.time())

    for gen in args.generators:
        for algo in args.algorithms:
            for i in range(args.runs):
                seed = seed_base + i
                res = run_single(
                    width=args.width,
                    height=args.height,
                    gen_method=gen,
                    algorithm=algo,
                    loop_percent=args.loop_percent,
                    num_rewards=args.num_rewards,
                    max_weight=args.max_weight,
                    max_ticks=20000,
                    seed=seed,
                )
                all_rows.append(res)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "raw_results.csv"), all_rows)

    summary = aggregate_results(all_rows)
    # Expand group columns so CSV is nice
    expanded = []
    for row in summary:
        gen, algo = row["group"]
        new_row = {k: v for k, v in row.items() if k != "group"}
        new_row["gen_method"] = gen
        new_row["algorithm"] = algo
        expanded.append(new_row)
    write_csv(os.path.join(args.out_dir, "summary.csv"), expanded)

    # Plot a few key metrics
    if HAS_MPL:
        for metric in [
            "elapsed_sec_avg",
            "steps_avg",
            "unique_explored_avg",
            "nodes_expanded_avg",
            "path_length_avg",
            "total_cost_avg",
            "frontier_max_avg",
        ]:
            plot_metric(expanded, metric, os.path.join(args.out_dir, f"{metric}.png"))

    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()


