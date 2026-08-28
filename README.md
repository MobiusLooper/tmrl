# RL Racer

A browser-based reinforcement-learning racing demo, built from scratch to make the environment and learning process easy to understand.

## Current status

Milestones 1 through 5 are complete. The project currently includes:

* a playable go-kart circuit with a chicane, hairpin and straights
* deterministic Python car physics and track collision detection
* keyboard driving through a TypeScript Canvas webapp
* lap timing, crash feedback and instant restarts
* isolated real-time sessions over FastAPI and WebSockets
* five live distance sensors rendered as rays with normalized numeric readings
* a reusable six-value observation containing sensor distances and speed
* exact, efficient sensor intersections against the track corridor geometry
* a deterministic `reset()` / `step()` environment with nine discrete actions
* ordered lap-progress checkpoints, configurable rewards and terminal outcomes
* a reusable headless episode runner and seeded random-policy baseline
* tabular Q-learning with configurable state discretisation and epsilon decay
* deterministic greedy evaluation isolated from the training random stream
* a headless training command with periodic learning metrics and JSON results
* automated tests for physics, geometry, sensors, environment, training and the browser-facing API

The seed-0 Q-learning validation reached 45% mean greedy-evaluation progress at episode 350, with a best checkpoint of 61%, compared with 2.609% mean progress for the 1,000-episode random baseline. Evaluation playback and the training dashboard follow in later milestones.

Evaluation progress during that run:

| Training episode | Epsilon | Mean progress | Best progress |
| ---: | ---: | ---: | ---: |
| 50 | 0.782 | 0% | 0% |
| 100 | 0.609 | 8% | 8% |
| 150 | 0.474 | 20% | 20% |
| 200 | 0.369 | 60.5% | 61% |
| 250 | 0.287 | 46.3% | 53% |
| 300 | 0.223 | 45% | 45% |
| 350 | 0.174 | 45% | 45% |

See [plan.md](plan.md) for the product and technical design, and [implementation-plan.md](implementation-plan.md) for the milestone roadmap.

## Run locally

Install the JavaScript dependencies once:

```bash
npm install
```

Then start the API and web client together:

```bash
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Drive with the arrow keys or WASD. Press `R` or Space to restart.

The five sensor rays cover relative angles `-60°`, `-30°`, `0°`, `+30°` and `+60°`. Their numeric readings are normalized over a 12-unit range, where `0` means the boundary is at the car and `1` means no boundary is detected within range.

Python dependencies are installed automatically into an isolated environment by `uv`.

## Test

```bash
npm test
```

## Random baseline

Run the deterministic 1,000-episode reference baseline headlessly:

```bash
uv run python -m backend.training.random_baseline --episodes 1000 --seed 0 --report-every 100
```

The command reports progress and finishes with aggregate reward, lap progress, terminal outcomes and throughput metrics. It does not retain frame-by-frame trajectories; browser playback of recorded evaluation runs is planned for Milestone 6.

## Q-learning

Train the tabular Q-learning agent and run a ten-episode greedy evaluation every 50 episodes:

```bash
uv run python -m backend.training.q_learning --episodes 1000 --seed 0 --evaluate-every 50 --evaluation-episodes 10 --report-every 50
```

The command reports training throughput, epsilon and evaluation progress, then prints the configuration, aggregate training summary, Q-table size and evaluation history as JSON when it finishes normally. Training and evaluation use separate seeded random streams, so changing evaluation frequency does not alter learning.

Current limitations:

* Metrics are printed to the terminal and the final JSON is not persisted automatically. Interrupting a run before completion loses its full metric history.
* The Q-table is memory-only. Interrupted training cannot yet resume from a checkpoint, although rerunning the same command and seed reproduces it deterministically.
* Evaluations currently retain aggregate metrics only. They do not record frame-by-frame trajectories, so learned policies cannot yet be played back in the browser.

Milestone 6 will add checkpoint persistence and resumption alongside recorded evaluation trajectories and browser playback.

## Production build

```bash
npm run build
npm start
```

The FastAPI server serves the built app at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Current browser API

* `GET /api/track` returns track geometry and authoritative sensor configuration.
* `/ws/play` creates an isolated manual-driving session and streams state snapshots containing the five normalized sensor readings.
