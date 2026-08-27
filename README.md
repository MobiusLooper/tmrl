# RL Racer

A browser-based reinforcement-learning racing demo, built from scratch to make the environment and learning process easy to understand.

## Current status

Milestones 1 and 2 are complete. The project currently includes:

* a playable go-kart circuit with a chicane, hairpin and straights
* deterministic Python car physics and track collision detection
* keyboard driving through a TypeScript Canvas webapp
* lap timing, crash feedback and instant restarts
* isolated real-time sessions over FastAPI and WebSockets
* five live distance sensors rendered as rays with normalized numeric readings
* a reusable six-value observation containing sensor distances and speed
* exact, efficient sensor intersections against the track corridor geometry
* automated tests for physics, geometry, sensors and the browser-facing API

Milestone 3, the reinforcement-learning environment API, is next. Learning and the training dashboard follow in later milestones.

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

## Production build

```bash
npm run build
npm start
```

The FastAPI server serves the built app at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Current browser API

* `GET /api/track` returns track geometry and authoritative sensor configuration.
* `/ws/play` creates an isolated manual-driving session and streams state snapshots containing the five normalized sensor readings.
