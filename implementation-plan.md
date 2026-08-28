# RL Racing Demo — Implementation Plan

This roadmap tracks delivery of the V1 design described in [plan.md](plan.md).

## Milestone 1 — Physics sandbox ✅

Implemented:

* reusable Python car and track simulation
* fixed kart circuit with straights, a chicane, hairpin and sweeping corners
* collision detection, crash handling and lap timing
* keyboard controls and a responsive Canvas renderer
* FastAPI and WebSocket integration for isolated browser sessions
* automated physics, geometry and API tests

Success criterion met:

> A human can intuitively drive, crash, restart and complete a timed lap in the webapp.

---

## Milestone 2 — Sensors ✅

Implemented:

* five egocentric raycasts with normalized boundary distances
* deterministic sensor and six-value observation helpers
* exact ray intersections with the centreline's capsule geometry
* live Canvas ray rendering while manually driving
* numeric sensor readout and browser API integration
* automated sensor geometry, reference-equivalence and API tests

The optimized sensor implementation reduced snapshot time from approximately 27.9 ms to 3.89 ms on the development machine.

Success criterion met:

> From the sensor values and speed alone, it appears plausible that the car has enough information to navigate the track.

---

## Milestone 3 — Environment API ✅

Implemented:

* deterministic `RacingEnv.reset()` and `step()` interface
* stable nine-action discrete action space
* equal-distance ordered checkpoints with forward and reverse progress
* additive step, checkpoint, crash and lap rewards
* crash, validated-lap and 1,200-step timeout termination
* typed observations, results and environment information
* focused determinism, reward, progress and termination tests

Success criterion met:

> Fixed action sequences produce deterministic observations, rewards, progress and termination while manual browser play remains unchanged.

---

## Milestone 4 — Random agent ✅

Implemented:

* minimal agent protocol and seeded uniformly random agent
* reusable synchronous episode runner
* deterministic per-episode records and aggregate run metrics
* headless random-baseline command with periodic throughput reporting
* reproducibility, terminal-outcome, runner and aggregation tests

Seed `0`, 1,000-episode baseline:

```text
total steps       323,906
steps/second      233.2
mean return       -10.55106
mean progress     2.609%
best progress     14%
crashes           992
timeouts          8
completed laps    0
```

Success criterion met:

> One command runs 1,000 deterministic random-policy episodes headlessly and produces the reference baseline that Q-learning must beat.

These episodes do not train an agent or retain trajectories. Learning begins in Milestone 5, and recorded browser playback follows in Milestone 6.

---

## Milestone 5 — Q-learning ✅

Implemented:

* six-value observation discretisation with configurable uniform buckets
* sparse nine-action Q-table
* terminal-aware Q-learning updates implemented without an RL library
* seeded epsilon-greedy exploration with episode-based decay
* independently seeded greedy evaluation that cannot mutate training
* reusable training and evaluation runners
* headless training command with periodic metrics and JSON results
* focused algorithm, determinism and real-environment integration tests

Seed `0`, 350-episode validation checkpoint:

```text
training steps              276,666
training steps/second       217.5
epsilon                     0.1739
latest evaluation progress  45%
best evaluation progress    61%
random mean progress        2.609%
```

Each evaluation used ten epsilon-zero episodes. The planned 1,000-episode run was stopped at episode 350 after the learned policy had already exceeded the random reference from episode 100 onward.

Success criterion met:

> Greedy evaluation performance is reproducibly and measurably better than random behaviour.

---

## Milestone 6 — Evaluation playback ✅

Implemented:

* versioned, human-readable JSON checkpoints with atomic replacement
* exact Q-table, metric-history and training-random-state restoration
* total-target resume semantics and graceful episode-boundary interruption
* best-run trajectory selection at every greedy evaluation checkpoint
* recorded physical states, sensors, actions, rewards and Q-values
* replay catalog, latest, per-checkpoint and lightweight all-trajectory HTTP endpoints
* shared manual/replay Canvas rendering with independent client-side timing
* pause, restart, scrub, checkpoint navigation and 1×/2×/5×/10× playback
* synchronized Compare All mode with early-blue-to-latest-lime trajectory overlays
* checkpoint, resumption, trajectory-selection and browser API tests

Success criterion:

> We can resume an interrupted run and watch how its current learnt policy drives.

---

## Milestone 7 — Full-lap learning architecture 🟡

Implemented:

* twelve-value continuous observations with seven sensor rays, progress, lateral offset and heading error
* pace- and crossing-speed checkpoint shaping with explicit crash, timeout and stall penalties
* near-wall-sensitive tabular discretisation and time-calibrated discounting
* transition-based epsilon decay, promotion reheating, and a seeded adaptive curriculum that expands backwards to canonical starts
* selectable tabular and locally implemented PyTorch Double DQN learners
* replay buffer, 10-step returns, target network, best-policy retention and resumable DQN state
* legacy replay compatibility and schema-2 DQN replay manifests
* a fresh `tabular-local-v5` path with an eight-part, 583,200-state upper bound based on local lateral and heading context rather than absolute progress sectors
* all nine tabular actions active at normal speed, including simultaneous brake-and-steer control
* 10 Hz tabular decisions over unchanged 20 Hz physics, with two-frame discounted updates
* deterministic sticky action selection for learned rows, safe unseen-state defaults, and low-speed propulsive action constraints
* 50% canonical curriculum sampling with bounded stage bands and a ten-second tabular stall limit
* architecture-aware checkpoint rejection for old Q-tables while retaining their browser replays
* canonical terminal-rate and replay steering-smoothness benchmark metrics

Implementation is complete. The DQN benchmark and a fresh Local-v5 acceptance run remain:

> A fresh seed-0 DQN run must produce a greedy canonical lap within 60 simulated seconds and one million training transitions.

> A fresh seed-0 Local-v5 run must exceed the previous 61% canonical best progress within 3,000 episodes, then complete a canonical lap within 60 simulated seconds while averaging no more than two steering-state changes per second in its best replay.

The fresh seed-0 Local-v4 run completed all 3,000 episodes with 68 exploratory training laps and zero stalls across 600 greedy evaluations. Its best greedy checkpoint reached 55% with 4.42 steering changes per second and no completed lap, so the liveness objective passed but the progress, lap, and smoothness acceptance criteria remain unmet.

Local-v5 enables simultaneous brake-and-steer actions and requires a fresh acceptance run; Local-v4 checkpoints remain replayable but cannot resume because their active-action semantics differ.

---

## Milestone 8 — Training dashboard

Add:

* episode count
* steps/sec
* epsilon
* reward graph
* lap-progress graph
* best lap
* current actions
* sensor readings
* Q-values

---

## Milestone 9 — Experiment controls

Expose RL and reward parameters.

Allow training to be reset and restarted with new values.

This turns the project from a demo into a small RL experimentation environment.
