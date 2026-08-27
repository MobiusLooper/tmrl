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

## Milestone 3 — Environment API

Next milestone.

Wrap the simulation behind:

```python
reset()
step(action)
```

Add:

* rewards
* checkpoints
* episode termination
* lap progress

Write tests for deterministic physics.

---

## Milestone 4 — Random agent

Create an agent that picks random actions.

Run thousands of episodes headlessly.

Record metrics.

This validates the full training loop before implementing learning.

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

Record evaluation trajectories.

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
