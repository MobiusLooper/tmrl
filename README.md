# RL Racer

A browser-based reinforcement-learning racing demo, built from scratch to make the environment and learning process easy to understand.

## Current status

Milestones 1 through 6 are complete. Milestone 7 is implemented and awaiting its full training benchmark. The project currently includes:

* a playable go-kart circuit with a chicane, hairpin and straights
* deterministic Python car physics and track collision detection
* keyboard driving through a TypeScript Canvas webapp
* lap timing, crash feedback and instant restarts
* isolated real-time sessions over FastAPI and WebSockets
* five live distance sensors rendered as rays with normalized numeric readings
* a ten-value observation containing sensor distances, speed and track-relative context
* exact, efficient sensor intersections against the track corridor geometry
* a deterministic `reset()` / `step()` environment with nine discrete actions
* ordered lap-progress checkpoints, configurable rewards and terminal outcomes
* a reusable headless episode runner and seeded random-policy baseline
* selectable smooth tabular Q-learning and continuous-observation Double DQN
* pace-aware rewards, stalled-episode termination and an adaptive backwards curriculum
* seven-action, 10 Hz tabular control with two-frame action persistence and sticky steering
* a 155,520-state tabular representation designed for state reuse
* deterministic greedy evaluation isolated from the training random stream
* atomic JSON checkpoints with exact deterministic training resumption
* recorded best-policy evaluation trajectories at every checkpoint
* browser replay with scrubbing, checkpoint sequences and 1×–10× playback
* automated tests for physics, geometry, sensors, environment, training and the browser-facing API

The original six-value seed-0 Q-learning validation reached 45% mean greedy-evaluation progress at episode 350, with a best checkpoint of 61%, compared with 2.609% mean progress for the 1,000-episode random baseline. That checkpoint remains replayable but cannot resume under the smooth tabular architecture. The full-lap benchmark for the new learner has not yet been run.

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

The shared learning observation adds normalized lap progress, signed lateral offset, and sine/cosine heading error. DQN consumes all ten continuous values. Smooth tabular learning deliberately uses only five discretised sensors, discretised speed, and one of four 25% progress sectors. Its five non-uniform sensor thresholds produce six bins per sensor, for an upper bound of 155,520 states. Ignoring lateral offset and heading error makes more driving situations share Q-values.

Rewards favour first-time checkpoint progress and early, moving checkpoint crossings; re-crossing an old checkpoint cannot farm positive reward. Crashes and 60-second timeouts retain explicit terminal penalties. Smooth tabular training and evaluation allow ten seconds without new progress, while DQN retains the environment's five-second default.

Smooth tabular control chooses among seven actions: coast, throttle, brake, left, left with throttle, right, and right with throttle. The two brake-and-steer enum values remain available to DQN and legacy replays but are never selected or used for tabular bootstrapping. A tabular decision is made at 10 Hz and held for two 50 ms physics frames. When the previous action is within `0.03` of the best Q-value it is retained, which supplies persistence without changing rewards. Replays still contain every 20 Hz physics frame and all nine browser-facing Q-values.

Tabular training uses a seeded backwards curriculum with 50% canonical starts. Other starts are sampled from the current 75–90%, 50–70%, or 25–45% band. Once at least 35% of the latest 50 non-canonical starts complete the remaining lap, training advances to the next band and reheats epsilon to `0.30`. Greedy evaluation always starts from the original grid at zero speed.

Tabular epsilon decays linearly by macro decisions rather than episodes or physics frames, so short curriculum episodes do not exhaust exploration prematurely. The default schedule moves epsilon from `1.0` to `0.10` over 200,000 decisions, approximately the same simulated driving as 400,000 former single-frame decisions. The terminal reports physical throughput, decision count, epsilon, canonical progress, lap completions, terminal rates, and replay steering changes.

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

The command reports progress and finishes with aggregate reward, lap progress, terminal outcomes and throughput metrics. The random baseline remains aggregate-only; trajectory recording is reserved for learnt-policy evaluation.

## Training

Run the enhanced tabular learner with ten-episode greedy evaluation every 50 episodes:

```bash
uv run python -m backend.training.train --algorithm tabular --episodes 3000 --seed 0
```

The exploration schedule can be changed with `--epsilon-decay-steps`, `--epsilon-min`, and `--epsilon-reheat`. The smooth-control defaults can be overridden with `--action-repeat`, `--sticky-tolerance`, `--canonical-start-probability`, and `--tabular-stall-seconds`.

Run the PyTorch Double DQN learner:

```bash
uv run python -m backend.training.train --algorithm dqn --episodes 3000 --max-transitions 1000000 --seed 0
```

Both learners evaluate from the canonical starting grid and preserve their best evaluated policy. Every fresh command creates a timestamped run under `artifacts/runs/`, for example `artifacts/runs/20260828T143012Z-tabular-smooth-seed0-a1b2c3d4/`. Tabular runs contain `checkpoint.json` and `best.json`; DQN runs contain `checkpoint.json`, `model.pt`, and `best-model.pt`.

The command prints its run ID and checkpoint path, reports training throughput, epsilon and evaluation progress, saves after every evaluation, and records the best greedy attempt from each evaluation batch. Training and evaluation use separate seeded random streams, so changing evaluation frequency does not alter learning.

Resume tabular training by supplying the checkpoint and a new total target episode:

```bash
uv run python -m backend.training.train --algorithm tabular \
  --resume artifacts/runs/RUN_ID/checkpoint.json --episodes 5000
```

Resume DQN training similarly:

```bash
uv run python -m backend.training.train --algorithm dqn \
  --resume artifacts/runs/RUN_ID/checkpoint.json --episodes 5000
```

The episode value is the new total target, not the number of additional episodes. Saved learner, curriculum and random state are restored, and the resumed checkpoint remains in its original run. `--checkpoint path/to/run.json` remains available for a fresh run with a custom output path, but cannot be combined with `--resume`.

Open the webapp and switch to **Agent Replay** to choose among discovered tabular and DQN runs, then watch one evaluation checkpoint or the entire sequence. Replays can be paused, scrubbed and played at 1×, 2×, 5× or 10× speed. The most recently updated run is selected by default. To pin a different default, including a checkpoint outside `artifacts/`, use:

```bash
RL_RACER_CHECKPOINT=path/to/checkpoint.json npm run dev
```

Legacy top-level checkpoints are included in the run selector and remain replayable. Six-part and previous nine-part tabular Q-tables cannot resume into the seven-part `tabular-smooth-v3` architecture; start a fresh run without `--resume`. Smooth-v3 checkpoints include the architecture, action repeat, sticky tolerance, curriculum probability, and stall duration needed for an exact compatible resume. Training runs from the command line; named experiment grouping, live metrics, and browser-controlled experiments remain future work.

## Production build

```bash
npm run build
npm start
```

The FastAPI server serves the built app at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Current browser API

* `GET /api/track` returns track geometry and authoritative sensor configuration.
* `GET /api/runs` returns discovered runs and the default run ID.
* `GET /api/runs/{run_id}/replays` returns one run's chronological evaluation catalog.
* `GET /api/runs/{run_id}/replays/latest` returns that run's newest selected trajectory.
* `GET /api/runs/{run_id}/replays/{training_episode}` returns one selected trajectory.
* The existing `GET /api/replays`, `/api/replays/latest`, and `/api/replays/{training_episode}` routes remain aliases for the default run.
* `/ws/play` creates an isolated manual-driving session and streams state snapshots containing the five normalized sensor readings.
