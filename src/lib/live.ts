import { useCallback, useEffect, useRef, useState } from 'react';

export type Mode = 'real' | 'hardwired' | 'shuffled';
export type SnakeState = { size: number; body: [number, number][]; food: [number, number]; score: number; alive: boolean; heading: number };
export type LiveFrame = {
  time: number; mode: Mode; episode: number; best: number; snake: SnakeState;
  channels: Record<string, number>; manual: boolean;
  action: 0 | 1 | 2; probabilities: [number, number, number];
  steer: Record<string, number>; activeNeurons: number; totalNeurons: number;
  values: [number, number][];
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
