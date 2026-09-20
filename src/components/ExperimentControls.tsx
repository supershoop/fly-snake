import { useEffect, useRef, useState } from 'react';
import type { Layout, LiveFrame, LiveStatus, PendingCommand, PolicyName, Wiring } from '../lib/live';
import { Icon } from './Icon';

const MAX_STEP_RATE = 5; // dashboard ceiling; the server independently clamps to its own range in Experiment.handle

const LAYOUTS: { layout: Layout; label: string; detail?: string; hint: string }[] = [
  { layout: 'solo', label: 'One fly', hint: 'One simulated brain. One snake. Follow the signal from senses to movement.' },
  { layout: 'swarm', label: 'Swarm', detail: '16', hint: 'Sixteen independent brains, one board each. Select a board to inspect its fly.' },
  { layout: 'versus', label: 'Versus', hint: 'You are the coral snake. Use arrow keys, WASD, or the direction buttons to play.' },
];

type Mode = 'trained' | 'normal' | 'scrambled' | 'training';
const MODES: { mode: Mode; label: string; message: object }[] = [
  { mode: 'trained', label: 'Trained', message: { wiring: 'real', policy: 'trained' } },
  { mode: 'normal', label: 'Normal', message: { wiring: 'real', policy: 'instinct' } },
  { mode: 'scrambled', label: 'Scrambled', message: { wiring: 'shuffled', policy: 'instinct' } },
  { mode: 'training', label: 'Training', message: { wiring: 'real', policy: 'learning' } },
];
const modeOf = (wiring: Wiring, policy: PolicyName): Mode => wiring === 'shuffled' ? 'scrambled' : policy === 'hardwired' || policy === 'instinct' ? 'normal' : policy === 'learning' ? 'training' : 'trained';

export function ExperimentControls({ frame, status, paused, pending, send }: {
  frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; send: (message: object) => void;
}) {
  const layout = pending?.layout ?? frame?.layout;
  const mode = frame && !frame.synaptic?.active ? modeOf(pending?.wiring ?? frame.wiring, pending?.policy ?? frame.policy) : null;
  const selectedMode = frame?.synaptic?.active ? 'synaptic' : mode ?? 'trained';
  const ready = status === 'live' && !!frame && !paused;

  // The slider sets a cap the host can move freely; the server reports what it actually
  // achieved (stepRate on the frame), since the brain can't be sped up past its compute time.
  const [stepRate, setStepRate] = useState(0); // 0 = uncapped, i.e. as fast as the brain computes
  const sendRate = useRef(send);
  sendRate.current = send;
  useEffect(() => {
    const timeout = setTimeout(() => sendRate.current({ stepRate }), 150); // debounce drag events
    return () => clearTimeout(timeout);
  }, [stepRate]);
  const achieved = frame?.stepRate;

  return <section className="experiment-controls" aria-label="Experiment settings">
    <div className="control-row">
      <div className="layout-control environment-control">
        <span className="eyebrow" id="layout-label">Environment</span>
        <div className="segmented" role="group" aria-labelledby="layout-label">
          {LAYOUTS.map(({ layout: option, label, detail, hint }) => <button key={option} disabled={!ready} title={hint} aria-pressed={layout === option} onClick={() => send({ layout: option })}>
            <Icon name={option}/><span>{label}</span>{detail && <small>{detail}</small>}
          </button>)}
        </div>
      </div>
      <div className="layout-control brain-control">
        <span className="eyebrow" id="mode-label">Brain</span>
        <div className="segmented">
          <select className="brain-select" aria-labelledby="mode-label" disabled={!ready} value={selectedMode} onChange={event => {
            const choice = MODES.find(option => option.mode === event.target.value);
            if (choice) send(choice.message);
          }}>
            {frame?.synaptic?.active && <option value="synaptic" disabled>Trained synapses · fixed readout</option>}
            {MODES.map(({ mode: option, label }) => <option key={option} value={option}>{label}</option>)}
          </select>
        </div>
      </div>
      <div className="layout-control step-rate-control">
        <span className="eyebrow" id="step-rate-label">Speed</span>
        <label htmlFor="step-rate" className="step-rate-slider">
          <input id="step-rate" type="range" min="0" max={MAX_STEP_RATE} step="0.25" disabled={!ready}
                 value={stepRate} onChange={event => setStepRate(Number(event.target.value))}
                 aria-labelledby="step-rate-label" title="Cap simulation moves per second; the brain can't run faster than it computes"/>
          {/* Fixed width so "Uncapped" vs "0.25/s" vs "3.00/s" never reflow the slider's track. */}
          <output htmlFor="step-rate">{stepRate === 0 ? 'Uncapped' : `${stepRate.toFixed(2)}/s`}</output>
        </label>
        <small className="step-rate-actual">{ready && achieved ? `${achieved.toFixed(2)} moves/s actual` : '\u00A0'}</small>
      </div>
    </div>
  </section>;
}
