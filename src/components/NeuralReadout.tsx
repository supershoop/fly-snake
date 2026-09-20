import { useCallback, useEffect, useRef, useState } from 'react';
import type { LiveFrame, LiveStatus, PendingCommand } from '../lib/live';

const INPUTS = [
  { name: 'food_L', label: 'Food left', type: 'LC10 L' },
  { name: 'food_R', label: 'Food right', type: 'LC10 R' },
  { name: 'danger_L', label: 'Threat left', type: 'LC4 L' },
  { name: 'danger_R', label: 'Threat right', type: 'LC4 R' },
  { name: 'danger_ahead', label: 'Threat ahead', type: 'LPLC2' },
];

export function NeuralReadout({ frame, status, paused, pending, send }: { frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; send: (message: object) => void }) {
  const [held, setHeld] = useState<string | null>(null);
  const active = useRef<string | null>(null);
  const fly = frame?.flies[frame.selected];
  const ready = status === 'live' && !!frame && !paused;
  const applyingInput = pending?.manual !== undefined;
  const release = useCallback(() => {
    if (active.current === null) return;
    active.current = null; setHeld(null); send({ stimulate: null });
  }, [send]);
  const hold = (name: string) => {
    if (!ready || active.current !== null) return;
    active.current = name; setHeld(name); send({ stimulate: { [name]: 1 } });
  };
  useEffect(() => {
    const hide = () => { if (document.hidden) release(); };
    window.addEventListener('blur', release);
    document.addEventListener('visibilitychange', hide);
    return () => { window.removeEventListener('blur', release); document.removeEventListener('visibilitychange', hide); release(); };
  }, [release]);
  useEffect(() => { if (!ready) release(); }, [ready, release]);
  const max = Math.max(50, ...Object.values(fly?.steer ?? {}));
  return <section className="signal-panel" aria-label="Brain output and sensory inputs">
    <div className="signal-block">
      <div className="block-heading"><h3>Descending neurons</h3><span>Hz · left / right</span></div>
      {['DNa02', 'DNa01', 'DNp01'].map(kind => <div className="neuron-row" key={kind}>
        <div><strong>{kind}</strong><span>{kind === 'DNp01' ? 'Escape' : 'Steering'}</span></div>
        <div className="paired-meters"><meter aria-label={`${kind} left firing rate`} min={0} max={max} value={fly?.steer[`${kind}_L`] ?? 0}/><meter aria-label={`${kind} right firing rate`} min={0} max={max} value={fly?.steer[`${kind}_R`] ?? 0}/></div>
        <span className="mono">{fly ? `${Math.round(fly.steer[`${kind}_L`] ?? 0)} / ${Math.round(fly.steer[`${kind}_R`] ?? 0)}` : '— / —'}</span>
      </div>)}
    </div>
    <div className="signal-block decision-block">
      <div className="block-heading"><h3>Readout decision</h3><span>Probability</span></div>
      {['Left', 'Straight', 'Right'].map((label, index) => <div className={`decision-row ${fly?.action === index ? 'chosen' : ''}`} key={label}>
        <span><span aria-hidden="true">{['↰', '↑', '↱'][index]}</span>{label}</span><meter aria-label={`${label} move probability`} min={0} max={1} value={fly?.probabilities[index] ?? 0}/><span className="mono">{fly ? `${Math.round(fly.probabilities[index] * 100)}%` : '—'}</span>
      </div>)}
      <p className="signal-note">{frame?.policy === 'hardwired' ? 'Steering neurons choose the turn. Nothing trained.' : 'A linear readout converts neural activity into a move.'}</p>
    </div>
    <div className="signal-block input-block">
      <div className="block-heading"><h3>Try a sensory input</h3><span className={held || applyingInput ? 'accent-text' : ''}>{held ? 'Override active' : applyingInput ? <><i className="spinner"/>{pending.message}</> : 'Press & hold'}</span></div>
      <div className="sensory-buttons">{INPUTS.map(({ name, label, type }) => <button key={name} disabled={!ready} aria-pressed={held === name} title={`Hold to stimulate ${type}. Release to return to the game.`}
        onPointerDown={event => { if (event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); hold(name); }}
        onPointerUp={release} onPointerCancel={release} onLostPointerCapture={release} onBlur={release}
        onClick={event => { if (event.detail === 0) { if (active.current === name) release(); else hold(name); } }}
        onKeyDown={event => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); if (!event.repeat) hold(name); } if (event.key === 'Escape') release(); }}
        onKeyUp={event => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); release(); } }}>
        <span>{label}</span><small>{type}</small>
      </button>)}</div>
      {frame?.manual && !held && <button className="release-override" disabled={!ready} onClick={() => send({ stimulate: null })}>Release sensory override</button>}
      <p className="signal-note">{paused ? 'Resume to stimulate the brain.' : 'Holding an input overrides the senses and holds the game still.'}</p>
    </div>
  </section>;
}
