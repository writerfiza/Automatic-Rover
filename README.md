# Autonomous Rover Navigation — AI-Based Terrain Perception and Path Planning

A software simulation of an autonomous planetary rover. Using only a simulated onboard camera, the rover classifies unknown terrain (free ground, dust, rocks, craters, steep slopes) with a trained AI model, then plans safe paths with cost-aware A* search. Tested against a rule-based baseline and a perfect-perception "Oracle" across 30+ randomized maps, the AI-assisted rover roughly doubles mission success rate and cuts collisions by about 7x compared to the baseline.

Built for the **World Space Week 2026 "Rocket Revolution"** theme and submitted to the **Stardust Challenge 2026**.

---

## What this project is

Real planetary rovers (Mars, Moon) cannot be joystick-controlled from Earth — radio signals take 3 to 22 minutes one way. A rover must interpret its surroundings and make short-term navigation decisions on its own. This project simulates that capability entirely in software: no physical hardware is used or required.

The rover follows a continuous loop:

```
OBSERVE → DETECT → DECIDE → NAVIGATE → (repeat)
```

- **Observe** — a simulated camera renders a noisy, hazy grayscale image of the terrain around the rover. The rover never sees the true map.
- **Detect** — a trained machine learning model (Random Forest) classifies each visible patch of ground.
- **Decide** — an A* path planner finds the safest route to the destination using only what the rover currently believes about the terrain, re-planning whenever new hazards are found.
- **Navigate** — the rover takes one step. If a hazard was missed, it registers a collision and updates its map with the truth.

## Where the AI is, specifically

The AI is the terrain classifier only — a Random Forest model trained with scikit-learn on five brightness/texture features extracted from each camera patch, labelled from ground-truth terrain across 30 training maps and evaluated on 10 entirely unseen test maps (never used in training). Path planning (A*) is a conventional algorithm by design — a common and defensible hybrid architecture (learned perception + classical planning), chosen because it is explainable and because reinforcement learning was judged beyond the project's three-week timeline.

**AI test accuracy: 87.7%** on unseen maps (training accuracy 91.5%, a small, healthy gap indicating limited overfitting). Full per-class results and confusion matrix are in `results/`.

## Key result

Three perception methods were run through an identical sense-plan-move loop, on the same 30 random seeded maps, so the only variable was perception quality:

| Method | Success rate | Avg. collisions per mission |
|---|---|---|
| Baseline (hand-written brightness rules) | 70.0% | 93.93 |
| **AI (trained Random Forest)** | **80.0%** | **13.37** |
| Oracle (perfect perception, upper bound) | 96.7% | 0.00 |

The AI-assisted rover reaches the goal more often and collides roughly **7x less often** than the non-AI baseline, while still falling short of the theoretical Oracle ceiling — a realistic, honestly measured result rather than a claim of perfection. See `results/comparison_chart.png`.

Stress testing (heavy dust, higher rock density, unseen seeds) shows the AI's advantage holds across conditions but narrows under haze beyond its training range — a genuine, stated limitation rather than a hidden one.

## Project structure

```
Rover_Project/
├── src/
│   ├── config.py              All settings and constants
│   ├── planet.py               Procedural terrain generation (rocks, craters, slopes, dust)
│   ├── rover.py                 Rover state, movement, collision handling
│   ├── camera.py                Simulated noisy/hazy camera
│   ├── planner.py               A* path planning (true-map "Oracle" cost function)
│   ├── perception_baseline.py   Rule-based (non-AI) terrain classifier
│   ├── perception_ai.py         Trained AI terrain classifier
│   ├── perception_oracle.py     Perfect-perception stand-in (upper bound)
│   ├── features.py              Image feature extraction for the AI model
│   ├── mission.py               The sense-plan-move autonomous loop
│   ├── generate_dataset.py      Builds the labelled training/test dataset
│   ├── train_model.py           Trains and evaluates the Random Forest classifier
│   ├── run_experiment.py        Baseline vs AI vs Oracle comparison (30 maps)
│   ├── run_stress_tests.py      Dust / rock-density / unseen-map stress tests
│   ├── demo_compare.py          Side-by-side Baseline vs AI visual demo
│   └── test_*.py                Unit tests for each component
├── results/                     Trained model, datasets, CSV results, charts
├── docs/                        Screenshots and supporting evidence
├── requirements.txt
└── README.md
```

## How to run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python src\generate_dataset.py   # build the training data
python src\train_model.py        # train the AI classifier
python src\run_experiment.py     # reproduce the Baseline vs AI vs Oracle comparison
python src\demo_compare.py 2026 0.25   # watch a live side-by-side mission
```

## What this simulation does and does not show

This is a 2D simulation with synthetic camera images, not real Mars/Moon imagery, and not a physical rover. It demonstrates that a trained perception model meaningfully improves autonomous navigation decisions over hand-written rules, under deliberately ambiguous, noisy sensing conditions. It does **not** demonstrate hardware reliability, real terrain imagery performance, or communication-delay handling — these are identified as future work.

## Future work

- Real or higher-fidelity terrain imagery
- Features that capture 3D shape (to address the model's weakest class, steep slopes)
- A small convolutional neural network, compared against the current classical model
- Reinforcement learning for the decision stage
- Physical rover prototype integration


## Author

Fiza Batool, 2nd-year Artificial Intelligence undergraduate.
Built for World Space Week 2026 ("Rocket Revolution") and the Stardust Challenge 2026.
