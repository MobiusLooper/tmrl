# RL Racing Demo — V1 Design

This document contains the product and technical design. Delivery order and implementation status are tracked separately in [implementation-plan.md](implementation-plan.md).

## 1. Goal

Build a small end-to-end reinforcement-learning demo in which an agent learns to drive a car around a fixed go-kart-style track.

The project should be simple enough that the important RL machinery is implemented from scratch and understandable, while still producing visibly interesting learning behaviour.

The finished app should let us:

1. Start training from scratch.
2. Watch training metrics improve over time.
3. Periodically watch the current policy attempt a lap.
4. See the car's sensor inputs and chosen actions.
5. Reset training and experiment with RL/reward parameters.
6. Understand the whole environment, training loop and learning algorithm without relying on an RL framework.

V1 is deliberately not a realistic driving simulator.

---

# 2. V1 Scope

## Included

* Fixed 2D go-kart-style track.
* Simple top-down car.
* Throttle.
* Brake.
* Left steering.
* Right steering.
* Constant speed when neither throttle nor brake is pressed.
* Constant angular steering rate, so higher speed naturally produces a wider turning circle.
* Collision with track boundary ends the episode.
* Agent observes simple simulated distance sensors plus current speed.
* Reinforcement learning implemented from scratch.
* Training runs in Python.
* Browser-based visualisation.
* Live training statistics.
* Periodic visual evaluation laps.
* Ability to reset and retrain.

## Explicitly excluded from V1

* Realistic tyre physics.
* Sliding/drifting.
* Lateral velocity.
* Weight transfer.
* Suspension.
* Gearbox.
* Engine torque curves.
* Damage.
* Other cars.
* Procedural tracks.
* Pixel/image observations.
* Continuous-action RL.
* Existing RL libraries such as Stable Baselines.

---

# 3. Overall Architecture

```text
┌──────────────────────────────┐
│         Web frontend         │
│                              │
│ Canvas track rendering       │
│ Car + sensors                │
│ Training graphs              │
│ Current episode              │
│ Hyperparameter controls      │
└──────────────┬───────────────┘
               │
          WebSocket / HTTP
               │
┌──────────────▼───────────────┐
│        Python server         │
│                              │
│ Training controller          │
│ Evaluation controller        │
│ Metrics/event streaming      │
└──────────────┬───────────────┘
               │
       ┌───────▼────────┐
       │ RL environment │
       │                │
       │ Car physics    │
       │ Track          │
       │ Sensors        │
       │ Rewards        │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │     Agent      │
       │                │
       │ Q-learning     │
       │ ε-greedy       │
       │ Q table        │
       └────────────────┘
```

Suggested stack:

* Python 3.12+
* FastAPI
* WebSockets
* TypeScript
* HTML Canvas
* Minimal frontend framework, or plain TypeScript
* No game engine
* No RL framework

---

# 4. Environment Interface

The environment should behave similarly to a stripped-down Gym environment, but be implemented ourselves.

```python
class RacingEnv:
    def reset(self) -> Observation:
        ...

    def step(self, action: Action) -> StepResult:
        ...
```

Each call to `step()` advances the simulation by a fixed timestep.

```python
@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict
```

Use a fixed timestep:

```text
dt = 0.05 seconds
```

Equivalent to 20 simulation steps per simulated second.

Training does not need to run in real time.

---

# 5. Car State

The complete physical state is:

```text
x
y
heading
speed
```

Where:

* `x`, `y`: world position
* `heading`: angle in radians
* `speed`: scalar forward velocity

The car always moves in the direction it is pointing.

There is no sideways velocity.

---

# 6. Controls

The logical controls are:

```text
THROTTLE
BRAKE
LEFT
RIGHT
```

Throttle/brake and steering may be active simultaneously.

Therefore valid action combinations include:

```text
coast
throttle
brake

left
left + throttle
left + brake

right
right + throttle
right + brake
```

This gives nine discrete actions.

Internally:

```python
@dataclass
class Action:
    throttle: bool
    brake: bool
    left: bool
    right: bool
```

Invalid combinations such as simultaneous throttle + brake or left + right do not need to exist in the discrete action space.

---

# 7. V1 Physics

## Longitudinal movement

Throttle increases speed at a constant acceleration.

```python
if throttle:
    speed += acceleration * dt
```

Brake reduces speed at a constant deceleration.

```python
if brake:
    speed -= braking * dt
```

If neither is active:

```python
speed = speed
```

There is deliberately:

* no drag
* no rolling resistance
* no automatic acceleration
* no automatic deceleration

Clamp speed:

```python
speed = clamp(speed, min_speed, max_speed)
```

For V1:

```text
min_speed = 0
```

The car can therefore stop completely.

Suggested initial values:

```text
acceleration = 4 world-units/s²
braking     = 7 world-units/s²
max_speed   = 12 world-units/s
```

These should eventually be tunable constants.

---

# 8. Steering Physics

Steering changes heading at a constant angular rate.

For example:

```python
if left:
    heading -= steering_rate * dt

if right:
    heading += steering_rate * dt
```

Suggested initial value:

```text
steering_rate = 1.4 rad/s
```

Crucially, steering rate does **not** change with speed.

Therefore the turning radius naturally becomes:

$$
r = \frac{v}{\omega}
$$

where:

* \(v\) = speed
* \(\omega\) = steering angular velocity

So:

```text
low speed  → tight turning radius
high speed → broad turning radius
```

This is exactly the behaviour wanted for V1 without introducing additional vehicle physics.

At zero speed, steering should have no effect.

So practically:

```python
if speed > 0:
    heading += steering_input * steering_rate * dt
```

---

# 9. Position Update

After acceleration/braking and steering:

```python
x += cos(heading) * speed * dt
y += sin(heading) * speed * dt
```

This is essentially the entire V1 vehicle simulation.

---

# 10. Car Geometry

Represent the car as a small oriented rectangle for rendering.

For collision detection, however, use a circle initially.

Example:

```text
collision radius = 0.4 world units
```

This keeps track collision code extremely simple.

The visual rectangle does not need to precisely correspond to collision geometry.

---

# 11. Track

Use one fixed go-kart-style circuit.

It should have:

* approximately 6–10 corners
* mix of left and right corners
* at least one hairpin
* one faster sweeping corner
* at least one left-right or right-left chicane
* one or two straights, including one relatively long straight
* consistent track width
* no intersections
* no elevation
* clearly defined start/finish line

Conceptually:

```text
             ╭────────────╮
        ╭────╯  straight  ╰──╮
        │                     │
   ╭────╯       ╭─╮ ╭─╮       │
   │             ╰╮╭╯         │
   │              ╰╯          │
   ╰──╮     chicane      ╭────╯
      ╰─────╯            ╰────
```

It should feel more like an indoor/outdoor kart track than a racing oval.

---

# 12. Track Representation

Represent the track using:

1. A centreline polyline.
2. A fixed track half-width.

For example:

```python
track_centerline = [
    (x1, y1),
    (x2, y2),
    ...
]
```

The actual drivable surface can be generated as a corridor around this centreline.

The centreline serves several purposes:

* rendering
* collision detection
* measuring lap progress
* reward calculation
* checkpoint generation

For V1, the centreline can be hand-authored.

---

# 13. Collision Detection

For each simulation step:

1. Determine the closest point on the track centreline to the car.
2. Calculate perpendicular distance from car to centreline.
3. If:

```text
distance + car_radius > track_half_width
```

the car has left the track.

Result:

```text
episode ends immediately
```

There is no bouncing or collision response in V1.

---

# 14. Progress Measurement

Generate ordered checkpoints along the track centreline.

For example:

```text
100 checkpoints around one lap
```

The environment tracks:

```text
current checkpoint
furthest valid checkpoint
lap progress
```

The car must pass checkpoints in order.

This prevents the agent from gaining reward by:

* driving backwards
* oscillating around one location
* cutting across nearby portions of the circuit

---

# 15. Agent Observations

The agent should not receive its absolute `(x, y)` position.

Instead, expose a small set of egocentric sensor readings.

## Distance sensors

Cast five rays from the car:

```text
-60°
-30°
  0°
+30°
+60°
```

Each returns distance to the nearest track boundary.

Use a maximum sensor range of:

```text
12 world units
```

Normalise each distance:

```text
0 = boundary immediately beside car
1 = no boundary within maximum sensor range
```

Visual representation:

```text
       \   |   /
        \  |  /
         \ | /
          🚗
```

## Speed

Include:

```text
speed / max_speed
```

Therefore the raw observation is:

```text
[
    ray_-60,
    ray_-30,
    ray_0,
    ray_30,
    ray_60,
    normalized_speed
]
```

Six values total.

This is intentionally minimal.

### Sensor geometry

The track corridor is the union of radius-`half_width` capsules around every centreline segment. Intersect each sensor ray directly with those capsules, merge their distance intervals, and use the end of the connected interval containing the car as the first boundary hit.

This exactly matches the collision geometry while avoiding iterative full-track distance scans. A sensor snapshot should remain comfortably below the 50 ms visual simulation timestep and fast enough to reuse in headless training.

---

# 16. Observation Discretisation

V1 should use tabular Q-learning.

Therefore continuous observations must be discretised.

For example:

Each sensor:

```text
5 buckets
```

Speed:

```text
5 buckets
```

This gives:

$$
5^6 = 15,625
$$

possible discrete states.

With 9 actions:

$$
15,625 × 9 = 140,625
$$

possible state/action values.

This is small enough to use a normal in-memory Q-table.

The exact bucket count can be tuned if learning is too slow or state representation too coarse.

---

# 17. RL Algorithm

Implement tabular Q-learning from scratch.

For transition:

```text
state
action
reward
next_state
```

update:

$$
Q(s,a) \leftarrow Q(s,a) +
\alpha
\left[
r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\right]
$$

No RL libraries.

Suggested starting parameters:

```text
learning rate α = 0.1
discount γ      = 0.99

initial ε       = 1.0
minimum ε       = 0.05
```

Exploration uses epsilon-greedy action selection.

---

# 18. Exploration Schedule

At the start:

```text
ε = 1.0
```

The car therefore acts almost entirely randomly.

Over training, epsilon decreases.

For example:

```text
ε = max(
    epsilon_min,
    epsilon_start * decay ** episode
)
```

The exact schedule should be visible in the UI.

An important part of the demo should be seeing the transition from:

```text
random exploration
```

to:

```text
mostly learned policy
```

---

# 19. Reward Function

Reward design should initially be very simple.

## Forward progress

Main reward:

```text
+ reward proportional to forward checkpoint progress
```

For example:

```text
+1 for each checkpoint advanced
```

## Backwards movement

```text
-1 for each checkpoint lost
```

or simply no positive reward.

Prefer initially penalising backwards movement slightly.

## Crash

```text
-10
```

## Completing a lap

```text
+50
```

## Time

Small per-step penalty:

```text
-0.01 per step
```

This discourages:

* stopping
* endlessly driving slowly
* taking unnecessarily long routes

The reward function should remain small enough that its behaviour is understandable.

---

# 20. Episode Termination

An episode ends if any of the following occur:

### Crash

Car leaves the track.

### Lap completed

Car crosses the finish line after completing all checkpoints.

### Timeout

Maximum simulation length exceeded.

Suggested initial timeout:

```text
60 simulated seconds
```

At `dt = 0.05`:

```text
1,200 environment steps
```

---

# 21. Starting State

At `reset()`:

* place car on starting grid
* align it with track
* set speed to zero
* reset checkpoint progress
* reset episode reward
* reset elapsed time

Optionally introduce a tiny random variation in initial position or heading later, but V1 should initially use an identical starting condition.

---

# 22. Training Loop

Conceptually:

```python
for episode in range(num_episodes):

    state = env.reset()
    done = False

    while not done:

        action = agent.choose_action(state)

        result = env.step(action)

        agent.update(
            state,
            action,
            result.reward,
            result.observation,
            result.done,
        )

        state = result.observation

    record_metrics()
```

Training should run as fast as possible.

It must not wait for frontend rendering.

---

# 23. Evaluation Episodes

Every N training episodes:

```text
e.g. every 50 episodes
```

run an evaluation.

During evaluation:

```text
ε = 0
```

The agent always selects its currently preferred action.

Do not update the Q-table during evaluation.

Record the complete trajectory:

```text
x
y
heading
speed
sensor values
chosen action
reward
```

for every timestep.

Send this trajectory to the frontend.

The browser can then play it back at approximately real-time speed.

This provides a stable visual representation of what the agent has actually learned.

---

# 24. Frontend

The main screen should contain three conceptual areas.

## Track view

Large Canvas showing:

* track
* centreline optionally
* start/finish
* checkpoints optionally
* car
* five sensor rays
* current evaluation trajectory

The car should clearly show orientation.

Sensor rays should terminate where they encounter a track edge.

---

# 25. Training Status

Show:

```text
Episode
Environment steps
Training speed (steps/sec)
Current epsilon
Current learning rate
Latest evaluation lap progress
Best lap progress
Best completed lap time
```

Example:

```text
Episode             1,284
Training speed      18,420 steps/sec
Exploration ε       0.17

Latest evaluation   73% lap
Best                 100%
Best lap             19.8 sec
```

---

# 26. Learning Graph

Plot episode or evaluation performance over time.

Initially show:

```text
evaluation lap progress vs episode
```

This is probably the single clearest metric.

Y-axis:

```text
0–100% lap
```

X-axis:

```text
training episode
```

A second graph can show total episode reward.

Do not overcomplicate V1 with many graphs.

---

# 27. Action Visualisation

During evaluation, display the current selected action.

For example:

```text
Throttle    ON
Brake       OFF
Left        ON
Right       OFF
```

Also show current speed.

Example:

```text
Speed: 8.4 / 12
```

This makes learnt braking behaviour visible.

---

# 28. Sensor Visualisation

Display the five ray values both:

* geometrically on the Canvas
* numerically in a small debug panel

For example:

```text
L60   0.31
L30   0.58
MID   0.91
R30   0.73
R60   0.22
```

This is useful for understanding what information the agent actually has.

---

# 29. Q-Value Visualisation

During evaluation, display the Q-value assigned to each possible action for the current discretised state.

Example:

```text
coast               2.1
throttle             5.8
brake               -0.7
left                  1.4
left + throttle       7.3  ← selected
left + brake          0.2
right                 1.8
right + throttle      2.9
right + brake        -1.1
```

This is one of the most educational parts of the application.

It makes the learned policy inspectable rather than treating the agent as a black box.

---

# 30. Controls

V1 UI controls:

```text
Start training
Pause training
Reset
```

Reset should:

* clear Q-table
* clear historical metrics
* reset epsilon
* restart from episode zero

Training parameters exposed in the UI:

```text
learning rate α
discount γ
epsilon decay
crash penalty
lap reward
step penalty
```

Changing parameters should require restarting training rather than mutating an existing run.

---

# 31. Training / Rendering Separation

The frontend must never limit training speed.

Training:

```text
runs as fast as possible
```

Frontend metrics:

```text
update perhaps 5 times / second
```

Evaluation trajectory:

```text
generated every N episodes
```

Browser:

```text
plays trajectory independently
```

For example:

```text
Python trains episodes 1,001–1,050

        ↓

Python performs evaluation #21

        ↓

trajectory streamed/sent to browser

        ↓

browser spends ~20 seconds playing it

Meanwhile:

Python is already training episodes 1,051+
```

The visualisation is therefore slightly behind training by design.

---

# 32. Server API

The manual-driving sandbox currently exposes:

```text
GET /api/track
WebSocket /ws/play
```

`GET /api/track` includes the track geometry plus sensor angles and maximum range. Each `/ws/play` state includes five normalized sensor readings ordered from `-60°` through `+60°`.

The training milestones add the following API:

## HTTP

```text
POST /training/start
POST /training/pause
POST /training/reset

GET /training/status
GET /config
POST /config
```

## WebSocket

```text
/ws/training
```

Events could include:

```text
training_metrics
evaluation_started
evaluation_trajectory
new_best
training_reset
```

Example metrics event:

```json
{
  "type": "training_metrics",
  "episode": 1284,
  "steps": 172920,
  "steps_per_second": 18420,
  "epsilon": 0.17,
  "mean_reward_100": 34.8,
  "latest_eval_progress": 0.73,
  "best_eval_progress": 1.0
}
```

---

# 33. Evaluation Trajectory Format

Example:

```json
{
  "episode": 1250,
  "frames": [
    {
      "t": 0.0,
      "x": 10.2,
      "y": 5.8,
      "heading": 1.73,
      "speed": 4.2,
      "sensors": [0.2, 0.4, 0.9, 0.7, 0.3],
      "action": "LEFT_THROTTLE",
      "q_values": [...]
    }
  ]
}
```

The frontend should animate these recorded frames rather than controlling the environment directly.

---

# 34. Suggested Project Structure

```text
rl-racer/
│
├── backend/
│   ├── env/
│   │   ├── environment.py
│   │   ├── car.py
│   │   ├── track.py
│   │   ├── sensors.py
│   │   └── geometry.py
│   │
│   ├── rl/
│   │   ├── agent.py
│   │   ├── q_table.py
│   │   └── discretisation.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   │
│   └── api/
│       └── server.py
│
├── frontend/
│   ├── src/
│   │   ├── trackRenderer.ts
│   │   ├── carRenderer.ts
│   │   ├── telemetry.ts
│   │   ├── charts.ts
│   │   └── app.ts
│
└── tests/
```

---

# 35. V1 Success Criteria

V1 is complete when:

1. A human can manually drive the car around the fixed track.

2. The car has throttle, brake, left and right controls.

3. Releasing throttle does not reduce speed.

4. Higher speed creates a larger turning circle purely because steering angular velocity is constant.

5. Leaving the track terminates the episode.

6. The agent receives only sensor distances and speed.

7. Tabular Q-learning is implemented without an RL library.

8. Starting from a fresh Q-table initially produces visibly poor/random driving.

9. Training eventually produces visibly better driving.

10. The UI shows learning progress quantitatively.

11. Periodic evaluation runs can be watched visually.

12. The UI shows sensor readings and action selection.

13. Training can run significantly faster than visual playback.

14. The user can reset training and experiment with learning/reward parameters.

---

# 36. What V1 Is Intended to Teach

By the end of V1, the implementation should make the following concepts concrete:

* environment
* state
* observation
* action
* reward
* episode
* return
* policy
* value function
* Q-value
* temporal-difference learning
* bootstrapping
* discount factor
* exploration vs exploitation
* epsilon-greedy policies
* state discretisation
* reward shaping
* evaluation vs training
* RL instability / sensitivity to parameters

The goal is not merely to make a car complete a lap. The system should make it possible to understand **why its behaviour changes as learning occurs**.

---

# 37. Natural V2

Once V1 works, the obvious next change is to keep exactly the same environment and replace:

```text
discretised observations
+
Q-table
```

with:

```text
continuous observations
+
small neural network
+
DQN
```

That means the project itself remains visually identical while the underlying learning method changes.

This gives a clean comparison:

```text
V1: tabular Q-learning
        ↓
V2: Deep Q Network
```

and introduces:

* neural-network function approximation
* replay buffer
* minibatch training
* target networks
* optimisation instability

without simultaneously changing the game.

That should be the first major extension after this V1 is working.
