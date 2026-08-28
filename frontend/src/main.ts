import "./style.css";

type Pair = [number, number];
type Gate = { start: Pair; end: Pair };
type Track = {
  centerline: Pair[];
  track_width: number;
  start_pose: { x: number; y: number; heading: number };
  finish_line: Gate;
  halfway_gate: Gate;
  sensors: { angles: number[]; max_range: number };
};
type RenderState = {
  tick: number;
  x: number;
  y: number;
  heading: number;
  speed: number;
  crashed: boolean;
  sensors: number[];
  current_progress: number;
};
type ManualSnapshot = RenderState & {
  type: "state";
  laps: number;
  current_lap_time: number;
  last_lap_time: number | null;
  best_lap_time: number | null;
  receivedAt: number;
};
type ReplayTransition = {
  action: number;
  action_name: string;
  q_values: number[];
  reward: number;
  state: RenderState;
};
type Replay = {
  training_episode: number;
  evaluation_episode: number;
  total_return: number;
  furthest_progress: number;
  simulated_duration: number;
  steps: number;
  termination_reason: "crash" | "lap" | "timeout" | "stalled";
  lap_completed: boolean;
  initial_state: RenderState;
  transitions: ReplayTransition[];
};
type ReplayMetadata = Omit<Replay, "initial_state" | "transitions">;
type ReplayCatalog = {
  run_id: string;
  schema_version: number;
  algorithm: "tabular" | "dqn" | string;
  latest_training_episode: number;
  replays: ReplayMetadata[];
};
type TrajectoryState = Pick<RenderState, "tick" | "x" | "y" | "heading" | "crashed">;
type Trajectory = ReplayMetadata & { states: TrajectoryState[] };
type TrajectoryCatalog = {
  run_id: string;
  latest_training_episode: number;
  trajectories: Trajectory[];
};
type TrainingRun = {
  run_id: string;
  created_at: string;
  updated_at: string;
  algorithm: "tabular" | "dqn" | string;
  seed: number;
  completed_episode: number;
  evaluation_count: number;
  latest_progress: number | null;
  best_progress: number | null;
  lap_completed: boolean;
};
type RunCatalog = {
  default_run_id: string | null;
  runs: TrainingRun[];
};
type Mode = "manual" | "replay";

const DT = 0.05;
const canvas = document.querySelector<HTMLCanvasElement>("#track")!;
const context = canvas.getContext("2d")!;
const shell = document.querySelector<HTMLElement>(".shell")!;
const speedLabel = document.querySelector<HTMLElement>("#speed-label")!;
const speedElement = document.querySelector<HTMLElement>("#speed")!;
const speedUnit = document.querySelector<HTMLElement>("#speed-unit")!;
const lapElement = document.querySelector<HTMLElement>("#lap")!;
const lapLabel = document.querySelector<HTMLElement>("#lap-label")!;
const timeElement = document.querySelector<HTMLElement>("#time")!;
const bestElement = document.querySelector<HTMLElement>("#best")!;
const bestLabel = document.querySelector<HTMLElement>("#best-label")!;
const connectionElement = document.querySelector<HTMLElement>("#connection")!;
const promptElement = document.querySelector<HTMLElement>("#prompt")!;
const crashElement = document.querySelector<HTMLElement>("#crash")!;
const toastElement = document.querySelector<HTMLElement>("#toast")!;
const restartButton = document.querySelector<HTMLButtonElement>("#restart")!;
const manualModeButton = document.querySelector<HTMLButtonElement>("#manual-mode")!;
const replayModeButton = document.querySelector<HTMLButtonElement>("#replay-mode")!;
const replayEmpty = document.querySelector<HTMLElement>("#replay-empty")!;
const replayEmptyDetail = document.querySelector<HTMLElement>("#replay-empty-detail")!;
const replaySummary = document.querySelector<HTMLElement>("#replay-summary")!;
const replaySummaryLabel = document.querySelector<HTMLElement>("#replay-summary-label")!;
const replayEpisode = document.querySelector<HTMLElement>("#replay-episode")!;
const replayResult = document.querySelector<HTMLElement>("#replay-result")!;
const replayControls = document.querySelector<HTMLElement>("#replay-controls")!;
const replayPlay = document.querySelector<HTMLButtonElement>("#replay-play")!;
const replayAll = document.querySelector<HTMLButtonElement>("#replay-all")!;
const replayPrevious = document.querySelector<HTMLButtonElement>("#replay-previous")!;
const replayNext = document.querySelector<HTMLButtonElement>("#replay-next")!;
const replayReload = document.querySelector<HTMLButtonElement>("#replay-reload")!;
const emptyReload = document.querySelector<HTMLButtonElement>("#empty-reload")!;
const replayTimeline = document.querySelector<HTMLInputElement>("#replay-timeline")!;
const replayRunSelect = document.querySelector<HTMLSelectElement>("#replay-run-select")!;
const replaySelect = document.querySelector<HTMLSelectElement>("#replay-select")!;
const replayRateSelect = document.querySelector<HTMLSelectElement>("#replay-rate")!;
const sensorElements = Array.from(document.querySelectorAll<HTMLElement>("[data-sensor]"));

const input = { throttle: false, brake: false, left: false, right: false };
let inputSequence = 0;
let socket: WebSocket | null = null;
let track: Track;
let mode: Mode = "manual";
let manualPrevious: ManualSnapshot | null = null;
let manualCurrent: ManualSnapshot | null = null;
let previousLapCount = 0;
let toastTimer = 0;
let catalog: ReplayCatalog | null = null;
let runCatalog: RunCatalog | null = null;
let selectedRunId: string | null = null;
let replay: Replay | null = null;
let trajectoryCatalog: TrajectoryCatalog | null = null;
let comparingAll = false;
let replayIndex = -1;
let replayPosition = 0;
let replayPlaying = false;
let replayRate = 1;
let replayLoading = false;
let lastAnimationTime = performance.now();

const formatTime = (seconds: number | null): string => {
  if (seconds === null) return "—";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes.toString().padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
};

const sendInput = (): void => {
  if (mode !== "manual" || socket?.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ type: "input", seq: ++inputSequence, ...input }));
};

const clearInput = (): void => {
  const changed = Object.values(input).some(Boolean);
  input.throttle = input.brake = input.left = input.right = false;
  if (changed && mode === "manual") sendInput();
};

const restartManual = (): void => {
  clearInput();
  promptElement.classList.add("hidden");
  crashElement.classList.remove("visible");
  socket?.send(JSON.stringify({ type: "reset" }));
};

const restartReplay = (): void => {
  if ((!comparingAll && !replay) || (comparingAll && !trajectoryCatalog)) return;
  replayPosition = 0;
  replayPlaying = true;
  lastAnimationTime = performance.now();
  updateReplayControls();
};

const keyToInput = (code: string): keyof typeof input | null => {
  if (code === "ArrowUp" || code === "KeyW") return "throttle";
  if (code === "ArrowDown" || code === "KeyS") return "brake";
  if (code === "ArrowLeft" || code === "KeyA") return "left";
  if (code === "ArrowRight" || code === "KeyD") return "right";
  return null;
};

window.addEventListener("keydown", (event) => {
  if (mode === "replay") {
    if ((event.code === "Space" || event.code === "KeyR") && !event.repeat) {
      event.preventDefault();
      if (event.code === "KeyR") restartReplay();
      else toggleReplay();
    }
    return;
  }
  const control = keyToInput(event.code);
  if (control) {
    event.preventDefault();
    if (!input[control]) {
      input[control] = true;
      promptElement.classList.add("hidden");
      sendInput();
    }
  } else if ((event.code === "KeyR" || event.code === "Space") && !event.repeat) {
    event.preventDefault();
    restartManual();
  }
});

window.addEventListener("keyup", (event) => {
  if (mode !== "manual") return;
  const control = keyToInput(event.code);
  if (control) {
    event.preventDefault();
    if (input[control]) {
      input[control] = false;
      sendInput();
    }
  }
});
window.addEventListener("blur", clearInput);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearInput();
    replayPlaying = false;
    updateReplayControls();
  }
});
restartButton.addEventListener("click", restartManual);

const connect = (): void => {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws/play`);
  connectionElement.className = "connection";
  connectionElement.lastChild!.textContent = " Connecting";

  socket.addEventListener("open", () => {
    connectionElement.classList.add("online");
    connectionElement.lastChild!.textContent = " Live";
    sendInput();
  });
  socket.addEventListener("message", (event) => {
    const snapshot = JSON.parse(event.data) as Omit<ManualSnapshot, "receivedAt" | "current_progress">;
    if (snapshot.type !== "state") return;
    manualPrevious = manualCurrent;
    manualCurrent = { ...snapshot, current_progress: 0, receivedAt: performance.now() };
    if (!manualPrevious) manualPrevious = manualCurrent;
    if (mode === "manual") updateManualHud(manualCurrent);
  });
  socket.addEventListener("close", () => {
    connectionElement.className = "connection offline";
    connectionElement.lastChild!.textContent = " Reconnecting";
    clearInput();
    window.setTimeout(connect, 1000);
  });
};

const updateManualHud = (state: ManualSnapshot): void => {
  speedLabel.textContent = "Speed";
  speedUnit.textContent = "/ 12";
  lapLabel.textContent = "Lap";
  bestLabel.textContent = "Best";
  speedElement.textContent = state.speed.toFixed(1);
  lapElement.textContent = String(state.laps + 1);
  timeElement.textContent = formatTime(state.current_lap_time);
  bestElement.textContent = formatTime(state.best_lap_time);
  crashElement.classList.toggle("visible", state.crashed);
  if (state.laps > previousLapCount && state.last_lap_time !== null) {
    toastElement.textContent = `LAP ${state.laps} · ${formatTime(state.last_lap_time)}`;
    toastElement.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toastElement.classList.remove("visible"), 2800);
  }
  previousLapCount = state.laps;
};

const updateSensors = (state: RenderState): void => {
  sensorElements.forEach((element) => {
    const panelIndex = Number(element.dataset.sensor);
    const index = state.sensors.length === 5 ? panelIndex - 1 : panelIndex;
    element.textContent = state.sensors[index]?.toFixed(2) ?? "—";
  });
};

const setMode = (nextMode: Mode): void => {
  if (nextMode === "replay" && mode === "manual") clearInput();
  mode = nextMode;
  shell.dataset.mode = mode;
  shell.dataset.review = comparingAll ? "compare" : "single";
  manualModeButton.classList.toggle("active", mode === "manual");
  replayModeButton.classList.toggle("active", mode === "replay");
  if (mode === "manual") {
    replayPlaying = false;
    if (manualCurrent) updateManualHud(manualCurrent);
  } else {
    crashElement.classList.remove("visible");
    promptElement.classList.add("hidden");
    if (!runCatalog) void loadRuns();
  }
  updateReplayControls();
};

manualModeButton.addEventListener("click", () => setMode("manual"));
replayModeButton.addEventListener("click", () => setMode("replay"));

const loadRuns = async (): Promise<void> => {
  replayLoading = true;
  showReplayError(null);
  updateReplayControls();
  try {
    const response = await fetch("/api/runs", { cache: "no-store" });
    if (!response.ok) throw new Error(await responseDetail(response));
    const previousRunId = selectedRunId;
    runCatalog = await response.json() as RunCatalog;
    replayRunSelect.replaceChildren(...runCatalog.runs.map((run) => {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = formatRun(run);
      return option;
    }));
    const selectedRun = runCatalog.runs.find((run) => run.run_id === previousRunId)
      ?? runCatalog.runs.find((run) => run.run_id === runCatalog?.default_run_id);
    if (!selectedRun) throw new Error("No training runs are available yet");
    selectedRunId = selectedRun.run_id;
    replayRunSelect.value = selectedRunId;
    await loadRunCatalog(selectedRunId);
  } catch (error) {
    runCatalog = null;
    selectedRunId = null;
    catalog = null;
    replay = null;
    trajectoryCatalog = null;
    comparingAll = false;
    showReplayError(error instanceof Error ? error.message : "Unable to load training runs");
  } finally {
    replayLoading = false;
    updateReplayControls();
  }
};

const loadRunCatalog = async (runId: string): Promise<void> => {
  replayLoading = true;
  replayPlaying = false;
  comparingAll = false;
  trajectoryCatalog = null;
  shell.dataset.review = "single";
  showReplayError(null);
  updateReplayControls();
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/replays`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(await responseDetail(response));
    catalog = await response.json() as ReplayCatalog;
    const compareOption = document.createElement("option");
    compareOption.value = "all";
    compareOption.textContent = "All checkpoints";
    replaySelect.replaceChildren(compareOption, ...catalog.replays.map((item) => {
      const option = document.createElement("option");
      option.value = String(item.training_episode);
      option.textContent = `Episode ${item.training_episode}`;
      return option;
    }));
    await loadReplay(catalog.replays.length - 1, false);
  } catch (error) {
    catalog = null;
    replay = null;
    showReplayError(error instanceof Error ? error.message : "Unable to load replay");
  } finally {
    replayLoading = false;
    updateReplayControls();
  }
};

const formatRun = (run: TrainingRun): string => {
  const timestamp = new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(run.created_at));
  const best = run.best_progress === null ? "no eval" : `${(run.best_progress * 100).toFixed(0)}% best`;
  return `${timestamp} · ${run.algorithm.toUpperCase()} · seed ${run.seed} · ep ${run.completed_episode} · ${best}`;
};

const loadReplay = async (index: number, keepPlaying: boolean): Promise<void> => {
  if (!catalog || index < 0 || index >= catalog.replays.length) return;
  replayLoading = true;
  updateReplayControls();
  const metadata = catalog.replays[index]!;
  try {
    if (!selectedRunId) throw new Error("No training run is selected");
    const response = await fetch(
      `/api/runs/${encodeURIComponent(selectedRunId)}/replays/${metadata.training_episode}`,
      { cache: "no-store" },
    );
    if (!response.ok) throw new Error(await responseDetail(response));
    replay = await response.json() as Replay;
    comparingAll = false;
    shell.dataset.review = "single";
    replayIndex = index;
    replayPosition = 0;
    replayPlaying = keepPlaying;
    replaySelect.value = String(replay.training_episode);
    replayTimeline.max = String(replay.steps);
    replayTimeline.value = "0";
    replaySummaryLabel.textContent = "Evaluation checkpoint";
    replayEpisode.textContent = `Episode ${replay.training_episode}`;
    replayResult.textContent = `${(replay.furthest_progress * 100).toFixed(0)}% progress · ${replay.termination_reason} · ${formatTime(replay.simulated_duration)}`;
    showReplayError(null);
    lastAnimationTime = performance.now();
  } catch (error) {
    replay = null;
    comparingAll = false;
    shell.dataset.review = "single";
    replayPlaying = false;
    showReplayError(error instanceof Error ? error.message : "Unable to load replay");
  } finally {
    replayLoading = false;
    updateReplayControls();
  }
};

const responseDetail = async (response: Response): Promise<string> => {
  try {
    const payload = await response.json() as { detail?: string };
    return payload.detail ?? `Replay request failed (${response.status})`;
  } catch {
    return `Replay request failed (${response.status})`;
  }
};

const showReplayError = (message: string | null): void => {
  replayEmpty.classList.toggle("visible", message !== null);
  replayEmptyDetail.textContent = message ?? "The latest checkpoint will appear here when an evaluation finishes.";
};

const toggleReplay = (): void => {
  const steps = playbackSteps();
  if (steps === 0) return;
  if (replayPosition >= steps) replayPosition = 0;
  replayPlaying = !replayPlaying;
  lastAnimationTime = performance.now();
  updateReplayControls();
};

const playbackSteps = (): number => {
  if (comparingAll) {
    return Math.max(0, ...(trajectoryCatalog?.trajectories.map((item) => item.steps) ?? []));
  }
  return replay?.steps ?? 0;
};

const showAllTrajectories = async (): Promise<void> => {
  if (!catalog?.replays.length || !selectedRunId) return;
  replayLoading = true;
  replayPlaying = false;
  showReplayError(null);
  updateReplayControls();
  try {
    if (!trajectoryCatalog) {
      const response = await fetch(
        `/api/runs/${encodeURIComponent(selectedRunId)}/trajectories`,
        { cache: "no-store" },
      );
      if (!response.ok) throw new Error(await responseDetail(response));
      trajectoryCatalog = await response.json() as TrajectoryCatalog;
    }
    comparingAll = true;
    shell.dataset.review = "compare";
    replayPosition = 0;
    replaySelect.value = "all";
    replayTimeline.max = String(playbackSteps());
    replayTimeline.value = "0";
    const trajectories = trajectoryCatalog.trajectories;
    replaySummaryLabel.textContent = "Learning run comparison";
    replayEpisode.textContent = "All checkpoints";
    replayResult.textContent = `Episodes ${trajectories[0]!.training_episode}–${trajectories.at(-1)!.training_episode} · early blue → latest lime`;
    lastAnimationTime = performance.now();
  } catch (error) {
    comparingAll = false;
    shell.dataset.review = "single";
    showReplayError(error instanceof Error ? error.message : "Unable to load trajectories");
  } finally {
    replayLoading = false;
    updateReplayControls();
  }
};

const updateReplayControls = (): void => {
  const available = (comparingAll ? trajectoryCatalog !== null : replay !== null) && !replayLoading;
  replayPlay.disabled = !available;
  replayRunSelect.disabled = replayLoading || !runCatalog?.runs.length;
  replayAll.disabled = !catalog?.replays.length || replayLoading;
  replayPrevious.disabled = comparingAll || !available || replayIndex <= 0;
  replayNext.disabled = comparingAll || !available || !catalog || replayIndex >= catalog.replays.length - 1;
  replayTimeline.disabled = !available;
  replaySelect.disabled = !available;
  replayPlay.textContent = replayPlaying ? "Pause" : "Play";
  replayAll.classList.toggle("active", comparingAll);
  replayControls.classList.toggle("loading", replayLoading);
  replaySummary.classList.toggle("visible", replay !== null || trajectoryCatalog !== null);
};

replayPlay.addEventListener("click", toggleReplay);
replayAll.addEventListener("click", () => {
  if (comparingAll) void loadReplay(replayIndex, false);
  else void showAllTrajectories();
});
replayPrevious.addEventListener("click", () => void loadReplay(replayIndex - 1, false));
replayNext.addEventListener("click", () => void loadReplay(replayIndex + 1, false));
replayReload.addEventListener("click", () => void loadRuns());
emptyReload.addEventListener("click", () => void loadRuns());
replayRunSelect.addEventListener("change", () => {
  selectedRunId = replayRunSelect.value;
  void loadRunCatalog(selectedRunId);
});
replaySelect.addEventListener("change", () => {
  if (replaySelect.value === "all") {
    void showAllTrajectories();
    return;
  }
  const index = catalog?.replays.findIndex((item) => item.training_episode === Number(replaySelect.value)) ?? -1;
  void loadReplay(index, false);
});
replayRateSelect.addEventListener("change", () => {
  replayRate = Number(replayRateSelect.value);
  lastAnimationTime = performance.now();
});
replayTimeline.addEventListener("input", () => {
  replayPlaying = false;
  replayPosition = Number(replayTimeline.value);
  updateReplayControls();
});

const replayRenderState = (now: number): RenderState | null => {
  if (!replay) return null;
  const elapsed = Math.max(0, now - lastAnimationTime) / 1000;
  lastAnimationTime = now;
  if (replayPlaying) replayPosition += elapsed / DT * replayRate;
  if (replayPosition >= replay.steps) {
    replayPosition = replay.steps;
    replayPlaying = false;
    updateReplayControls();
  }
  replayTimeline.value = String(Math.floor(replayPosition));
  const lower = Math.floor(replayPosition);
  const upper = Math.min(replay.steps, lower + 1);
  const from = replayStateAt(lower);
  const to = replayStateAt(upper);
  const state = interpolateState(from, to, replayPosition - lower);
  updateReplayHud(state);
  return state;
};

const replayStateAt = (frame: number): RenderState => {
  if (!replay) throw new Error("No replay is loaded");
  if (frame <= 0) return replay.initial_state;
  return replay.transitions[Math.min(frame, replay.steps) - 1]!.state;
};

const updateReplayHud = (state: RenderState): void => {
  if (!replay) return;
  speedLabel.textContent = "Speed";
  speedUnit.textContent = "/ 12";
  lapLabel.textContent = "Progress";
  bestLabel.textContent = "Return";
  speedElement.textContent = state.speed.toFixed(1);
  lapElement.textContent = `${(state.current_progress * 100).toFixed(0)}%`;
  timeElement.textContent = formatTime(replayPosition * DT);
  bestElement.textContent = replay.total_return.toFixed(1);
};

type PositionedTrajectory = {
  trajectory: Trajectory;
  state: TrajectoryState;
  index: number;
  finished: boolean;
};

const comparisonRenderStates = (now: number): PositionedTrajectory[] => {
  if (!trajectoryCatalog) return [];
  const elapsed = Math.max(0, now - lastAnimationTime) / 1000;
  lastAnimationTime = now;
  const maximum = playbackSteps();
  if (replayPlaying) replayPosition += elapsed / DT * replayRate;
  if (replayPosition >= maximum) {
    replayPosition = maximum;
    replayPlaying = false;
    updateReplayControls();
  }
  replayTimeline.value = String(Math.floor(replayPosition));
  const positioned = trajectoryCatalog.trajectories.map((trajectory, index) => {
    const position = Math.min(replayPosition, trajectory.steps);
    const lower = Math.floor(position);
    const upper = Math.min(trajectory.steps, lower + 1);
    return {
      trajectory,
      index,
      state: interpolateTrajectoryState(
        trajectory.states[lower]!,
        trajectory.states[upper]!,
        position - lower,
      ),
      finished: replayPosition >= trajectory.steps,
    };
  });
  updateComparisonHud(positioned);
  return positioned;
};

const interpolateTrajectoryState = (
  from: TrajectoryState,
  to: TrajectoryState,
  amount: number,
): TrajectoryState => ({
  ...to,
  x: from.x + (to.x - from.x) * amount,
  y: from.y + (to.y - from.y) * amount,
  heading: from.heading + (to.heading - from.heading) * amount,
});

const updateComparisonHud = (positioned: PositionedTrajectory[]): void => {
  speedLabel.textContent = "Checkpoints";
  speedUnit.textContent = "";
  lapLabel.textContent = "Active";
  bestLabel.textContent = "Laps";
  speedElement.textContent = String(positioned.length);
  lapElement.textContent = String(positioned.filter((item) => !item.finished).length);
  timeElement.textContent = formatTime(replayPosition * DT);
  bestElement.textContent = String(
    positioned.filter((item) => item.finished && item.trajectory.lap_completed).length,
  );
};

const interpolateState = (from: RenderState, to: RenderState, amount: number): RenderState => ({
  ...to,
  x: from.x + (to.x - from.x) * amount,
  y: from.y + (to.y - from.y) * amount,
  heading: from.heading + (to.heading - from.heading) * amount,
  speed: from.speed + (to.speed - from.speed) * amount,
  current_progress: from.current_progress + (to.current_progress - from.current_progress) * amount,
  sensors: to.sensors.map((value, index) => {
    const previousValue = from.sensors[index] ?? value;
    return previousValue + (value - previousValue) * amount;
  }),
});

const drawGate = (gate: Gate, color: string, dashed = false): void => {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = 0.16;
  if (dashed) context.setLineDash([0.35, 0.25]);
  context.beginPath();
  context.moveTo(...gate.start);
  context.lineTo(...gate.end);
  context.stroke();
  context.restore();
};

const traceCenterline = (): void => {
  context.beginPath();
  context.moveTo(...track.centerline[0]!);
  for (const point of track.centerline.slice(1)) context.lineTo(...point);
  context.closePath();
};

const drawSensors = (car: RenderState): void => {
  context.save();
  context.lineWidth = 0.065;
  context.strokeStyle = "rgba(186, 244, 60, .72)";
  context.fillStyle = "#dfff9c";
  const angles = car.sensors.length === 5
    ? [-60, -30, 0, 30, 60]
    : track.sensors.angles;
  for (const [index, relativeDegrees] of angles.entries()) {
    const reading = car.sensors[index] ?? 1;
    const distance = reading * track.sensors.max_range;
    const angle = car.heading + relativeDegrees * Math.PI / 180;
    const endX = car.x + Math.cos(angle) * distance;
    const endY = car.y + Math.sin(angle) * distance;
    context.beginPath();
    context.moveTo(car.x, car.y);
    context.lineTo(endX, endY);
    context.stroke();
    if (reading < 0.9999) {
      context.beginPath();
      context.arc(endX, endY, 0.105, 0, Math.PI * 2);
      context.fill();
    }
  }
  context.restore();
};

const trajectoryColor = (index: number, count: number, alpha = 1): string => {
  const amount = count <= 1 ? 1 : index / (count - 1);
  const hue = 205 + (82 - 205) * amount;
  const lightness = 58 + 8 * amount;
  return `hsla(${hue}, 82%, ${lightness}%, ${alpha})`;
};

const drawComparison = (positioned: PositionedTrajectory[]): void => {
  const count = positioned.length;
  for (const { trajectory, index } of positioned) {
    const latest = index === count - 1;
    context.beginPath();
    context.moveTo(trajectory.states[0]!.x, trajectory.states[0]!.y);
    for (const state of trajectory.states.slice(1)) context.lineTo(state.x, state.y);
    context.strokeStyle = trajectoryColor(index, count, latest ? 0.82 : 0.2 + index / count * 0.22);
    context.lineWidth = latest ? 0.17 : 0.085;
    context.stroke();
  }
  for (const { state, index, finished } of positioned) {
    const latest = index === count - 1;
    drawCar(
      state,
      trajectoryColor(index, count),
      finished ? (latest ? 0.52 : 0.25) : (latest ? 1 : 0.7),
      latest ? 0.9 : 0.65,
    );
  }
};

const drawCar = (
  car: Pick<RenderState, "x" | "y" | "heading" | "crashed">,
  color: string,
  alpha = 1,
  size = 1,
): void => {
  context.save();
  context.globalAlpha = alpha;
  context.translate(car.x, car.y);
  context.rotate(car.heading);
  context.scale(size, size);
  context.fillStyle = car.crashed ? "#ff6258" : color;
  context.strokeStyle = "#0b0f0c";
  context.lineWidth = 0.08;
  context.beginPath();
  context.roundRect(-0.62, -0.34, 1.24, 0.68, 0.16);
  context.fill();
  context.stroke();
  context.fillStyle = "#182019";
  context.fillRect(0.16, -0.24, 0.26, 0.48);
  context.fillStyle = "#f3f7ec";
  context.beginPath();
  context.moveTo(0.63, 0);
  context.lineTo(0.44, 0.15);
  context.lineTo(0.44, -0.15);
  context.closePath();
  context.fill();
  context.restore();
};

const render = (now: number): void => {
  requestAnimationFrame(render);
  if (!track) return;
  let car: RenderState | null = null;
  let compared: PositionedTrajectory[] = [];
  if (mode === "replay") {
    if (comparingAll) compared = comparisonRenderStates(now);
    else car = replayRenderState(now) ?? manualCurrent;
  } else if (manualCurrent && manualPrevious) {
    const alpha = Math.min(1, Math.max(0, (now - manualCurrent.receivedAt) / (DT * 1000)));
    car = interpolateState(manualPrevious, manualCurrent, alpha);
    lastAnimationTime = now;
  }
  if (!car && compared.length === 0) return;
  if (car) updateSensors(car);

  const bounds = track.centerline.reduce(
    (acc, [x, y]) => ({ minX: Math.min(acc.minX, x), maxX: Math.max(acc.maxX, x), minY: Math.min(acc.minY, y), maxY: Math.max(acc.maxY, y) }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity },
  );
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#131c16";
  context.fillRect(0, 0, width, height);

  const padding = Math.max(28, Math.min(width, height) * 0.07);
  const worldWidth = bounds.maxX - bounds.minX + track.track_width;
  const worldHeight = bounds.maxY - bounds.minY + track.track_width;
  const scale = Math.min((width - padding * 2) / worldWidth, (height - padding * 2) / worldHeight);
  const contentWidth = (bounds.maxX - bounds.minX) * scale;
  const contentHeight = (bounds.maxY - bounds.minY) * scale;
  const offsetX = (width - contentWidth) / 2 - bounds.minX * scale;
  const offsetY = (height - contentHeight) / 2 + bounds.maxY * scale;
  context.setTransform(scale * dpr, 0, 0, -scale * dpr, offsetX * dpr, offsetY * dpr);

  context.lineJoin = "round";
  context.lineCap = "round";
  traceCenterline();
  context.strokeStyle = "#070a08";
  context.lineWidth = track.track_width + 0.42;
  context.stroke();
  traceCenterline();
  context.strokeStyle = "#303934";
  context.lineWidth = track.track_width;
  context.stroke();
  traceCenterline();
  context.strokeStyle = "rgba(216, 229, 214, .22)";
  context.lineWidth = 0.055;
  context.setLineDash([0.45, 0.6]);
  context.stroke();
  context.setLineDash([]);

  drawGate(track.halfway_gate, "rgba(186,244,60,.24)", true);
  drawGate(track.finish_line, "#edf4e7");
  if (compared.length) {
    drawComparison(compared);
  } else if (car) {
    drawSensors(car);
    drawCar(car, "#baf43c");
  }
};

const boot = async (): Promise<void> => {
  const response = await fetch("/api/track");
  if (!response.ok) throw new Error(`Unable to load track (${response.status})`);
  track = await response.json() as Track;
  shell.dataset.mode = "manual";
  connect();
  requestAnimationFrame(render);
};

boot().catch((error: unknown) => {
  connectionElement.className = "connection offline";
  connectionElement.lastChild!.textContent = " Failed to start";
  console.error(error);
});
