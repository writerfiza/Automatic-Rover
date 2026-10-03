import numpy as np
import csv
from planet import generate_planet
from perception_baseline import observe_with_baseline
from perception_ai import observe_with_ai
from mission import run_mission
import config

STRESS_SEEDS = list(range(4000, 4020))   # 20 more unseen maps

SCENARIOS = {
    "Normal (haze 0.25)": {"haze": 0.25, "rock_fraction": None},
    "Heavy dust (haze 0.7)": {"haze": 0.7, "rock_fraction": None},
    "Rockier terrain (15% rocks)": {"haze": 0.25, "rock_fraction": 0.15},
}

METHODS = {
    "Baseline": observe_with_baseline,
    "AI": observe_with_ai,
}


def run_scenario(scenario_name, settings):
    rows = []
    original_rock_fraction = config.ROCK_FRACTION
    if settings["rock_fraction"] is not None:
        config.ROCK_FRACTION = settings["rock_fraction"]

    for method_name, perceive_fn in METHODS.items():
        for seed in STRESS_SEEDS:
            terrain, elevation = generate_planet(seed)
            result = run_mission(terrain, elevation, perceive_fn, settings["haze"],
                                  rng=np.random.default_rng(seed))
            rows.append({
                "scenario": scenario_name,
                "method": method_name,
                "seed": seed,
                "reached_goal": result["outcome"] == "reached_goal",
                "collisions": result["collisions"],
            })

    config.ROCK_FRACTION = original_rock_fraction
    return rows


if __name__ == "__main__":
    all_rows = []
    for scenario_name, settings in SCENARIOS.items():
        print(f"\nRunning scenario: {scenario_name}")
        all_rows.extend(run_scenario(scenario_name, settings))

    print("\n" + "=" * 65)
    print("STRESS TEST SUMMARY")
    print("=" * 65)
    for scenario_name in SCENARIOS:
        print(f"\n{scenario_name}")
        for method_name in METHODS:
            subset = [r for r in all_rows
                      if r["scenario"] == scenario_name and r["method"] == method_name]
            success = 100 * sum(r["reached_goal"] for r in subset) / len(subset)
            avg_coll = np.mean([r["collisions"] for r in subset])
            print(f"  {method_name:10s}  success: {success:5.1f}%   avg collisions: {avg_coll:5.2f}")

    with open("results/stress_test_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nSaved to results/stress_test_results.csv")