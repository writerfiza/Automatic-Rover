import numpy as np
import csv
from planet import generate_planet
from perception_baseline import observe_with_baseline
from perception_ai import observe_with_ai
from perception_oracle import observe_with_oracle
from mission import run_mission

EXPERIMENT_SEEDS = list(range(3000, 3030))   # 30 fresh, unseen test maps
HAZE_LEVEL = 0.25


def oracle_wrapper(terrain):
    def fn(photo, rover_pos, view_range):
        return observe_with_oracle(photo, rover_pos, view_range, terrain=terrain)
    return fn


METHODS = {
    "Baseline (rule-based)": lambda terrain: observe_with_baseline,
    "AI (Random Forest)": lambda terrain: observe_with_ai,
    "Oracle (perfect)": oracle_wrapper,
}


def run_all():
    rows = []
    for method_name, make_perceive_fn in METHODS.items():
        print(f"\nRunning {method_name}...")
        for seed in EXPERIMENT_SEEDS:
            terrain, elevation = generate_planet(seed)
            perceive_fn = make_perceive_fn(terrain)
            result = run_mission(terrain, elevation, perceive_fn, HAZE_LEVEL,
                                  rng=np.random.default_rng(seed))
            rows.append({
                "method": method_name,
                "seed": seed,
                "outcome": result["outcome"],
                "reached_goal": result["outcome"] == "reached_goal",
                "steps": result["steps"],
                "collisions": result["collisions"],
                "replans": result["replans"],
                "distance_m": result["distance_m"],
                "dust_crossings": result["dust_crossings"],
            })
    return rows


def print_summary(rows):
    print("\n" + "=" * 60)
    print("SUMMARY (averaged over", len(EXPERIMENT_SEEDS), "maps)")
    print("=" * 60)
    for method_name in METHODS:
        method_rows = [r for r in rows if r["method"] == method_name]
        success_rate = 100 * sum(r["reached_goal"] for r in method_rows) / len(method_rows)
        avg_collisions = np.mean([r["collisions"] for r in method_rows])
        avg_replans = np.mean([r["replans"] for r in method_rows])
        avg_distance = np.mean([r["distance_m"] for r in method_rows if r["reached_goal"]]) \
            if any(r["reached_goal"] for r in method_rows) else float("nan")
        print(f"\n{method_name}")
        print(f"  Success rate:        {success_rate:.1f}%")
        print(f"  Avg collisions:      {avg_collisions:.2f}")
        print(f"  Avg replans:         {avg_replans:.2f}")
        print(f"  Avg distance (successful runs): {avg_distance:.1f} m")


if __name__ == "__main__":
    rows = run_all()
    print_summary(rows)

    with open("results/experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved detailed results to results/experiment_results.csv")