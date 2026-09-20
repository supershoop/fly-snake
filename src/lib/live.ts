import { useCallback, useEffect, useRef, useState } from 'react';

export type Layout = 'solo' | 'swarm' | 'versus' | 'arena';
export type PolicyName = 'trained' | 'hardwired' | 'instinct' | 'learning';
export type Wiring = 'real' | 'shuffled';
export type SnakeState = { kind: 'fly' | 'human'; body: [number, number][]; heading: number; alive: boolean; score: number; games: number; lastScore: number };
export type ArenaState = { size: number; foods: [number, number][]; snakes: SnakeState[] };
export type FlyState = {
  // danger_* includes immediate collision and losing the route to the moving tail.
  arena: number; snake: number; channels: Record<string, number>; action: 0 | 1 | 2; probabilities: [number, number, number];
  reward: number; steer: Record<string, number>; lesion: string[]; feedbackEligible?: boolean;
};
export type FeedbackState = {
  positive: number; negative: number;
  last: { status: 'applied'; value: number; fly: number | null; move: number; targets: number[] }
    | { status: 'rejected'; reason: string } | null;
};
/** Server -> client, one per move. Documented in AGENTS.md; keep both in sync. */
export type LiveFrame = {
  time: number; move?: number; layout: Layout; wiring: Wiring; policy: PolicyName; manual: boolean; sensor: Record<string, number>;
  arenas: ArenaState[]; flies: FlyState[]; selected: number; lesionPresets: string[];
  learning: { moves: number; games: number; scores: number[]; feedback?: FeedbackState };
  activeNeurons: number; totalNeurons: number; values: [number, number][];
  /** bodyIds of the shown fly's silenced cells that the atlas draws, and how many cells are silenced in total. */
  encoder?: 'channels' | 'retina';
  /** Selected fly: firing rate (Hz) of each pathway node, and the retina cells lit this move as [cell index, drive 0..1]. */
  vision?: { pathway: Record<string, number>; view: [number, number][] };
  silenced?: number[]; silencedTotal?: number;
  /** fly index -> bodyIds of its silenced, drawn cells; only lesioned flies appear. */
  silencedByFly?: Record<string, number[]>;
};
/** [type, number of cells, superclass] for every annotated neuron type; requested once with {hello: true}. */
export type NeuronType = [string, number, string];
/** One-off host catalogue and reachable phone controller links, returned by {hello: true}. */
export type HelloMessage = { hello: { types: NeuronType[]; feedbackUrls?: string[] } };
/** Feedback-only phone socket at /feedback/ws; it shares the host's experiment. */
export type AudienceCommand = { id: string; feedback: number; fly: number | null; move: number };
export type AudienceFrame = Pick<LiveFrame, 'policy' | 'manual' | 'selected' | 'arenas' | 'flies'> & {
  move: number; paused: boolean; feedback: Pick<FeedbackState, 'positive' | 'negative'>;
};
export type AudienceMessage = { frame: AudienceFrame } | { id: string | null; receipt: NonNullable<FeedbackState['last']> };
/** Private authenticated socket, separate from host frames and audience feedback. */
export type OperatorPreset = 'crash' | 'untrained' | 'legacy' | 'evolved' | 'best';
export type OperatorCommand = { key: string } | { status: true } | { preset: OperatorPreset | 'release' };
export type OperatorState = {
  presets: { id: OperatorPreset; label: string; description: string }[];
  active: OperatorPreset | null; supported: boolean; mode: PolicyName | null;
  paused: boolean; move: number; reason: string;
};
export type OperatorMessage = { state: OperatorState } | { result: { ok: true; state: OperatorState } | { ok: false; reason: string } };
/** Sent once with the hello reply. Built by flybrain/vision.py from the connectome. */
export type VisionStatic = {
  pathway: { nodes: { id: string; label: string; side: 'L' | 'R'; role: string; bodyIds: number[] }[]; edges: [string, string][] };
  /** Eye columns as [u, v, side]: u = hex1 - hex2 (large = front of the eye), v = hex1 + hex2 (large = dorsal). */
  eye: { columns: [number, number, 'L' | 'R'][] };
  /** Retina cells as [azimuth deg (negative = left), u, v, side, isFoodDetector]. */
  retina: { cells: [number, number, number, 'L' | 'R', boolean][]; fieldDeg: [number, number]; threatRange: number };
};
export type LiveStatus = 'connecting' | 'live' | 'offline';
export type PendingCommand = {
  message: string;
  /** Human steering is a continuous play action, not a blocking UI operation. */
  kind?: 'human-move';
  layout?: Layout;
  policy?: PolicyName;
  wiring?: Wiring;
  selected?: number;
  manual?: boolean;
  framesRemaining?: number;
};

const url = () => import.meta.env.VITE_BRAIN_WS ?? `ws://${location.hostname || 'localhost'}:8000/ws`;

const layoutNames: Record<Layout, string> = { solo: 'one fly', swarm: 'the swarm', versus: 'human vs fly', arena: 'the shared arena' };
const policyNames: Record<PolicyName, string> = { trained: 'the trained readout', hardwired: 'nothing-trained mode', instinct: 'the untrained brain', learning: 'live learning' };

function isLayout(value: unknown): value is Layout {
  return value === 'solo' || value === 'swarm' || value === 'versus' || value === 'arena';
}

function isPolicy(value: unknown): value is PolicyName {
  return value === 'trained' || value === 'hardwired' || value === 'instinct' || value === 'learning';
}

function isWiring(value: unknown): value is Wiring {
  return value === 'real' || value === 'shuffled';
}

function describeCommand(message: Record<string, unknown>): PendingCommand | null {
  if (isLayout(message.layout)) return { message: `Switching to ${layoutNames[message.layout]}…`, layout: message.layout };
  if (isPolicy(message.policy) && isWiring(message.wiring)) return { message: message.wiring === 'shuffled' ? 'Switching to scrambled wiring…' : `Switching to ${policyNames[message.policy]}…`, policy: message.policy, wiring: message.wiring };
  if (isPolicy(message.policy)) return { message: `Switching to ${policyNames[message.policy]}…`, policy: message.policy };
  if (isWiring(message.wiring)) return { message: `Switching to ${message.wiring === 'real' ? 'real' : 'scrambled'} wiring…`, wiring: message.wiring };
  if (typeof message.select === 'number') return { message: `Selecting fly ${message.select + 1}…`, selected: message.select };
  if ('learning' in message) return { message: 'Preparing live learning…', policy: 'learning' };
  if ('lesion' in message) return { message: 'Applying the lesion…' };
  if ('feedback' in message) return { message: 'Applying feedback…' };
  if ('stimulate' in message) return { message: message.stimulate === null ? 'Returning to the game…' : 'Applying the sensory input…', manual: message.stimulate !== null };
  if ('human' in message) return { message: 'Sending your move…', kind: 'human-move' };
  if ('sensor' in message) return { message: 'Updating the sensor input…' };
  if ('paused' in message && message.paused === false) return { message: 'Resuming the simulation…' };
  return null;
}

function commandIsConfirmed(frame: LiveFrame, pending: PendingCommand) {
  return (pending.layout === undefined || frame.layout === pending.layout)
    && (pending.policy === undefined || frame.policy === pending.policy)
    && (pending.wiring === undefined || frame.wiring === pending.wiring)
    && (pending.selected === undefined || frame.selected === pending.selected)
    && (pending.manual === undefined || frame.manual === pending.manual);
}

function hasExpectedState(update: PendingCommand) {
  return update.layout !== undefined || update.policy !== undefined || update.wiring !== undefined || update.selected !== undefined || update.manual !== undefined;
}

/** One stimulus clock: every frame carries the game state and the brain activity it produced. Stale activity is cleared on disconnect. */
export function useLiveBrain() {
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [status, setStatus] = useState<LiveStatus>('connecting');
  const [types, setTypes] = useState<NeuronType[]>([]);
  const [vision, setVision] = useState<VisionStatic | null>(null);
  const [feedbackUrls, setFeedbackUrls] = useState<string[]>([]);
  const [pending, setPending] = useState<PendingCommand | null>(null);
  const asked = useRef(false);
  const socket = useRef<WebSocket | null>(null);
  useEffect(() => {
    let closed = false, retry = 0;
    const connect = () => {
      const ws = new WebSocket(url());
      socket.current = ws;
      ws.onopen = () => setStatus('live');
      ws.onmessage = event => {
        const message = JSON.parse(event.data);
        if (message.hello) { setTypes(message.hello.types); setVision(message.hello.vision ?? null); setFeedbackUrls(message.hello.feedbackUrls ?? []); return; }
        if (!asked.current) { asked.current = true; ws.send(JSON.stringify({ hello: true })); }
        setFrame(message);
        setPending(current => {
          if (!current || !commandIsConfirmed(message, current)) return current;
          if (current.framesRemaining && current.framesRemaining > 1) return { ...current, framesRemaining: current.framesRemaining - 1 };
          return null;
        });
      };
      ws.onclose = () => { if (closed) return; asked.current = false; setStatus('offline'); setFrame(null); setPending(null); setFeedbackUrls([]); retry = window.setTimeout(connect, 1500); };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); socket.current?.close(); };
  }, []);
  const send = useCallback((message: object) => {
    if (socket.current?.readyState !== WebSocket.OPEN) return;
    const update = describeCommand(message as Record<string, unknown>);
    if (update) setPending(current => {
      const next = hasExpectedState(update) ? update : { ...update, framesRemaining: 2 };
      if (!update.layout || !current) return { ...current, ...next };
      const { selected: _discardedSelection, ...beforeLayout } = current;
      return { ...beforeLayout, ...next };
    });
    socket.current.send(JSON.stringify(message));
  }, []);
  return { frame, status, send, types, pending, feedbackUrls, vision };
}
