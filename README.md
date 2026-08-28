# RL Racer

A browser-based reinforcement-learning racing demo, built from scratch to make the environment and learning process easy to understand.

## Current status

Milestones 1 through 4 are complete. The project currently includes:

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
* automated tests for physics, geometry, sensors, environment, training and the browser-facing API

Milestone 5, tabular Q-learning, is next. There is not yet a trained agent: the current agent chooses every action randomly and retains no knowledge between episodes. Evaluation playback and the training dashboard follow in later milestones.

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

## Production build

```bash
npm run build
npm start
```

The FastAPI server serves the built app at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Current browser API

* `GET /api/track` returns track geometry and authoritative sensor configuration.
* `/ws/play` creates an isolated manual-driving session and streams state snapshots containing the five normalized sensor readings.
