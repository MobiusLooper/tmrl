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

## Milestone 5 — Q-learning

Implement:

* state discretisation
* Q-table
* Q-learning update
* epsilon-greedy exploration
* epsilon decay

Success criterion:

> Evaluation performance becomes measurably better than random behaviour.

---

## Milestone 6 — Evaluation playback

Record selected evaluation trajectories from the current policy.

Send them to the frontend.

Animate them on Canvas.

Success criterion:

> We can watch how the current learnt policy drives.

---

## Milestone 7 — Training dashboard

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

## Milestone 8 — Experiment controls

Expose RL and reward parameters.

Allow training to be reset and restarted with new values.

This turns the project from a demo into a small RL experimentation environment.
