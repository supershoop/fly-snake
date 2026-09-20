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
  silenced?: number[]; silencedTotal?: number;
  /** fly index -> bodyIds of its silenced, drawn cells; only lesioned flies appear. */
  silencedByFly?: Record<string, number[]>;
};
/** [type, number of cells, superclass] for every annotated neuron type; requested once with {hello: true}. */
export type NeuronType = [string, number, string];
export type LiveStatus = 'connecting' | 'live' | 'offline';

const url = () => import.meta.env.VITE_BRAIN_WS ?? `ws://${location.hostname || 'localhost'}:8000/ws`;

/** One stimulus clock: every frame carries the game state and the brain activity it produced. Stale activity is cleared on disconnect. */
export function useLiveBrain() {
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [status, setStatus] = useState<LiveStatus>('connecting');
  const [types, setTypes] = useState<NeuronType[]>([]);
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
        if (message.hello) { setTypes(message.hello.types); return; }
        if (!asked.current) { asked.current = true; ws.send(JSON.stringify({ hello: true })); }
        setFrame(message);
      };
      ws.onclose = () => { if (closed) return; asked.current = false; setStatus('offline'); setFrame(null); retry = window.setTimeout(connect, 1500); };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); socket.current?.close(); };
  }, []);
  const send = useCallback((message: object) => { if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify(message)); }, []);
  return { frame, status, send, types };
}
