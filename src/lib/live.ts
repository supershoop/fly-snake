import { useCallback, useEffect, useRef, useState } from 'react';

export type Layout = 'solo' | 'swarm' | 'versus' | 'arena';
export type PolicyName = 'trained' | 'hardwired' | 'learning';
export type Wiring = 'real' | 'shuffled';
export type SnakeState = { kind: 'fly' | 'human'; body: [number, number][]; heading: number; alive: boolean; score: number; games: number; lastScore: number };
export type ArenaState = { size: number; foods: [number, number][]; snakes: SnakeState[] };
export type FlyState = {
  arena: number; snake: number; channels: Record<string, number>; action: 0 | 1 | 2; probabilities: [number, number, number];
  reward: number; steer: Record<string, number>; lesion: string[];
};
/** Server -> client, one per move. Documented in AGENTS.md; keep both in sync. */
export type LiveFrame = {
  time: number; layout: Layout; wiring: Wiring; policy: PolicyName; manual: boolean; sensor: Record<string, number>;
  arenas: ArenaState[]; flies: FlyState[]; selected: number; lesionPresets: string[];
  learning: { moves: number; games: number; scores: number[] };
  activeNeurons: number; totalNeurons: number; values: [number, number][];
};
export type LiveStatus = 'connecting' | 'live' | 'offline';

const url = () => import.meta.env.VITE_BRAIN_WS ?? `ws://${location.hostname || 'localhost'}:8000/ws`;

/** One stimulus clock: every frame carries the game state and the brain activity it produced. Stale activity is cleared on disconnect. */
export function useLiveBrain() {
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [status, setStatus] = useState<LiveStatus>('connecting');
  const socket = useRef<WebSocket | null>(null);
  useEffect(() => {
    let closed = false, retry = 0;
    const connect = () => {
      const ws = new WebSocket(url());
      socket.current = ws;
      ws.onopen = () => setStatus('live');
      ws.onmessage = event => setFrame(JSON.parse(event.data));
      ws.onclose = () => { if (closed) return; setStatus('offline'); setFrame(null); retry = window.setTimeout(connect, 1500); };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); socket.current?.close(); };
  }, []);
  const send = useCallback((message: object) => { if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify(message)); }, []);
  return { frame, status, send };
}
