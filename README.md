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
* selectable enhanced tabular Q-learning and continuous-observation Double DQN
* pace-aware rewards, stalled-episode termination and an adaptive backwards curriculum
* deterministic greedy evaluation isolated from the training random stream
* atomic JSON checkpoints with exact deterministic training resumption
* recorded best-policy evaluation trajectories at every checkpoint
* browser replay with scrubbing, checkpoint sequences and 1×–10× playback
* automated tests for physics, geometry, sensors, environment, training and the browser-facing API

The original six-value seed-0 Q-learning validation reached 45% mean greedy-evaluation progress at episode 350, with a best checkpoint of 61%, compared with 2.609% mean progress for the 1,000-episode random baseline. That checkpoint remains replayable but cannot resume under the new observation architecture. The full-lap benchmark for the new learners has not yet been run.

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

The learning observation adds normalized lap progress, signed lateral offset, and sine/cosine heading error. Rewards favour first-time checkpoint progress and early, moving checkpoint crossings; re-crossing an old checkpoint cannot farm positive reward. Crashes, 60-second timeouts, and five seconds without new progress all carry explicit terminal penalties. A discount of `0.9995` is calibrated to the simulator's 20 Hz decision rate.

Training uses a seeded backwards curriculum: most early episodes begin on later track checkpoints, while canonical starts remain in the mix. Once recent completion performance is sufficient, the starting floor moves from 75% to 50%, 25%, and finally canonical-only training. Greedy evaluation always starts from the original grid at zero speed.

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

Run the PyTorch Double DQN learner:

```bash
uv run python -m backend.training.train --algorithm dqn --episodes 3000 --max-transitions 1000000 --seed 0
```

Both learners evaluate from the canonical starting grid and preserve their best evaluated policy. Tabular training defaults to `artifacts/latest.json` and `artifacts/latest-best.json`. DQN defaults to `artifacts/dqn-latest.json`, with resumable `dqn-latest.pt` and best-policy `dqn-latest-best.pt` sidecars.

The command reports training throughput, epsilon and evaluation progress, saves `artifacts/latest.json` after every evaluation, and records the best greedy attempt from each evaluation batch. Training and evaluation use separate seeded random streams, so changing evaluation frequency does not alter learning.

Resume tabular training by supplying the checkpoint and a new total target episode:

```bash
uv run python -m backend.training.train --algorithm tabular --resume artifacts/latest.json --episodes 5000
```

Resume DQN training similarly:

```bash
uv run python -m backend.training.train --algorithm dqn --resume artifacts/dqn-latest.json --episodes 5000
```

The episode value is the new total target, not the number of additional episodes. Saved learner, curriculum and random state are restored. Use `--checkpoint path/to/run.json` to keep experiments separate.

Open the webapp and switch to **Agent Replay** to watch one selected checkpoint or the entire training sequence. Replays can be paused, scrubbed and played at 1×, 2×, 5× or 10× speed. The server reads `artifacts/latest.json` by default; point it at a different run with:

```bash
RL_RACER_CHECKPOINT=path/to/run.json npm run dev
```

Legacy six-value checkpoints remain replayable, but cannot resume into the ten-value learning architecture. To train without replacing the legacy artifact, pass a new path such as `--checkpoint artifacts/tabular-v2.json`. Training still runs from the command line; live metrics and browser-controlled experiments remain future work.

## Production build

```bash
npm run build
npm start
```

The FastAPI server serves the built app at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Current browser API

* `GET /api/track` returns track geometry and authoritative sensor configuration.
* `GET /api/replays` returns the chronological evaluation replay catalog.
* `GET /api/replays/latest` returns the newest selected trajectory.
* `GET /api/replays/{training_episode}` returns one selected checkpoint trajectory.
* `/ws/play` creates an isolated manual-driving session and streams state snapshots containing the five normalized sensor readings.
