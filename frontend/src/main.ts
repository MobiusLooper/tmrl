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
type Snapshot = {
  type: "state";
  tick: number;
  x: number;
  y: number;
  heading: number;
  speed: number;
  crashed: boolean;
  laps: number;
  current_lap_time: number;
  last_lap_time: number | null;
  best_lap_time: number | null;
  sensors: number[];
  receivedAt: number;
};

const canvas = document.querySelector<HTMLCanvasElement>("#track")!;
const context = canvas.getContext("2d")!;
const speedElement = document.querySelector<HTMLElement>("#speed")!;
const lapElement = document.querySelector<HTMLElement>("#lap")!;
const timeElement = document.querySelector<HTMLElement>("#time")!;
const bestElement = document.querySelector<HTMLElement>("#best")!;
const connectionElement = document.querySelector<HTMLElement>("#connection")!;
const promptElement = document.querySelector<HTMLElement>("#prompt")!;
const crashElement = document.querySelector<HTMLElement>("#crash")!;
const toastElement = document.querySelector<HTMLElement>("#toast")!;
const restartButton = document.querySelector<HTMLButtonElement>("#restart")!;
const sensorElements = Array.from(document.querySelectorAll<HTMLElement>("[data-sensor]"));

const input = { throttle: false, brake: false, left: false, right: false };
let inputSequence = 0;
let socket: WebSocket | null = null;
let track: Track;
let previous: Snapshot | null = null;
let current: Snapshot | null = null;
let previousLapCount = 0;
let toastTimer = 0;

const formatTime = (seconds: number | null): string => {
  if (seconds === null) return "—";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes.toString().padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
};

const sendInput = (): void => {
  if (socket?.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ type: "input", seq: ++inputSequence, ...input }));
};

const clearInput = (): void => {
  const changed = Object.values(input).some(Boolean);
  input.throttle = input.brake = input.left = input.right = false;
  if (changed) sendInput();
};

const restart = (): void => {
  clearInput();
  promptElement.classList.add("hidden");
  crashElement.classList.remove("visible");
  socket?.send(JSON.stringify({ type: "reset" }));
};

const keyToInput = (code: string): keyof typeof input | null => {
  if (code === "ArrowUp" || code === "KeyW") return "throttle";
  if (code === "ArrowDown" || code === "KeyS") return "brake";
  if (code === "ArrowLeft" || code === "KeyA") return "left";
  if (code === "ArrowRight" || code === "KeyD") return "right";
  return null;
};

window.addEventListener("keydown", (event) => {
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
    restart();
  }
});

window.addEventListener("keyup", (event) => {
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
document.addEventListener("visibilitychange", () => document.hidden && clearInput());
restartButton.addEventListener("click", restart);

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
    const snapshot = JSON.parse(event.data) as Omit<Snapshot, "receivedAt">;
    if (snapshot.type !== "state") return;
    previous = current;
    current = { ...snapshot, receivedAt: performance.now() };
    if (!previous) previous = current;
    updateHud(current);
  });
  socket.addEventListener("close", () => {
    connectionElement.className = "connection offline";
    connectionElement.lastChild!.textContent = " Reconnecting";
    clearInput();
    window.setTimeout(connect, 1000);
  });
};

const updateHud = (state: Snapshot): void => {
  speedElement.textContent = state.speed.toFixed(1);
  lapElement.textContent = String(state.laps + 1);
  timeElement.textContent = formatTime(state.current_lap_time);
  bestElement.textContent = formatTime(state.best_lap_time);
  crashElement.classList.toggle("visible", state.crashed);
  sensorElements.forEach((element) => {
    const index = Number(element.dataset.sensor);
    element.textContent = state.sensors[index]?.toFixed(2) ?? "—";
  });

  if (state.laps > previousLapCount && state.last_lap_time !== null) {
    toastElement.textContent = `LAP ${state.laps} · ${formatTime(state.last_lap_time)}`;
    toastElement.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toastElement.classList.remove("visible"), 2800);
  }
  previousLapCount = state.laps;
};

const interpolate = (from: Snapshot, to: Snapshot, amount: number): Snapshot => ({
  ...to,
  x: from.x + (to.x - from.x) * amount,
  y: from.y + (to.y - from.y) * amount,
  heading: from.heading + (to.heading - from.heading) * amount,
  speed: from.speed + (to.speed - from.speed) * amount,
  sensors: to.sensors.map((value, index) => {
    const previousValue = from.sensors[index] ?? value;
    return previousValue + (value - previousValue) * amount;
  }),
});

const drawGate = (gate: Gate, scale: number, color: string, dashed = false): void => {
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
  context.moveTo(...track.centerline[0]);
  for (const point of track.centerline.slice(1)) context.lineTo(...point);
  context.closePath();
};

const drawSensors = (car: Snapshot): void => {
  context.save();
  context.lineWidth = 0.065;
  context.strokeStyle = "rgba(186, 244, 60, .72)";
  context.fillStyle = "#dfff9c";
  for (const [index, relativeDegrees] of track.sensors.angles.entries()) {
    const distance = (car.sensors[index] ?? 1) * track.sensors.max_range;
    const angle = car.heading + relativeDegrees * Math.PI / 180;
    const endX = car.x + Math.cos(angle) * distance;
    const endY = car.y + Math.sin(angle) * distance;
    context.beginPath();
    context.moveTo(car.x, car.y);
    context.lineTo(endX, endY);
    context.stroke();
    if ((car.sensors[index] ?? 1) < 0.9999) {
      context.beginPath();
      context.arc(endX, endY, 0.105, 0, Math.PI * 2);
      context.fill();
    }
  }
  context.restore();
};

const render = (now: number): void => {
  requestAnimationFrame(render);
  if (!track || !current || !previous) return;

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

  drawGate(track.halfway_gate, scale, "rgba(186,244,60,.24)", true);
  drawGate(track.finish_line, scale, "#edf4e7");

  const alpha = Math.min(1, Math.max(0, (now - current.receivedAt) / 50));
  const car = interpolate(previous, current, alpha);
  drawSensors(car);
  context.save();
  context.translate(car.x, car.y);
  context.rotate(car.heading);
  context.fillStyle = car.crashed ? "#ff6258" : "#baf43c";
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

const boot = async (): Promise<void> => {
  const response = await fetch("/api/track");
  if (!response.ok) throw new Error(`Unable to load track (${response.status})`);
  track = await response.json() as Track;
  connect();
  requestAnimationFrame(render);
};

boot().catch((error: unknown) => {
  connectionElement.className = "connection offline";
  connectionElement.lastChild!.textContent = " Failed to start";
  console.error(error);
});
