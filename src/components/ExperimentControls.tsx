import type { Layout, LiveFrame, LiveStatus, PendingCommand, PolicyName, Wiring } from '../lib/live';
import { Icon } from './Icon';

const LAYOUTS: { layout: Layout; label: string; detail?: string; hint: string }[] = [
  { layout: 'solo', label: 'One fly', hint: 'One simulated brain. One snake. Follow the signal from senses to movement.' },
  { layout: 'swarm', label: 'Swarm', detail: '16', hint: 'Sixteen independent brains, one board each. Select a board to inspect its fly.' },
  { layout: 'versus', label: 'Versus', hint: 'You are the coral snake. Use arrow keys, WASD, or the direction buttons to play.' },
];

type Mode = 'trained' | 'normal' | 'scrambled' | 'training';
const MODES: { mode: Mode; label: string; hint: string; message: object }[] = [
  { mode: 'trained', label: 'Trained', hint: 'Real wiring. A linear readout of the descending neurons, fitted offline, picks the move.', message: { wiring: 'real', policy: 'trained' } },
  { mode: 'normal', label: 'Normal', hint: 'Real wiring, nothing trained. Steering neurons (DNa02, DNa01) pull the snake toward food; the giant fiber (DNp01) vetoes turns into a threat and triggers dodges.', message: { wiring: 'real', policy: 'instinct' } },
  { mode: 'scrambled', label: 'Scrambled', hint: 'Control. Same neurons and synapse strengths, random targets, nothing trained.', message: { wiring: 'shuffled', policy: 'instinct' } },
  { mode: 'training', label: 'Training', hint: 'Real wiring. The readout learns while playing, from reward alone.', message: { wiring: 'real', policy: 'learning' } },
];
const modeOf = (wiring: Wiring, policy: PolicyName): Mode => wiring === 'shuffled' ? 'scrambled' : policy === 'hardwired' || policy === 'instinct' ? 'normal' : policy === 'learning' ? 'training' : 'trained';

export function ExperimentControls({ frame, status, paused, pending, send }: {
  frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; send: (message: object) => void;
}) {
  const layout = pending?.layout ?? frame?.layout;
  const mode = frame ? modeOf(pending?.wiring ?? frame.wiring, pending?.policy ?? frame.policy) : null;
  const selectedMode = mode ?? 'trained';
  const ready = status === 'live' && !!frame && !paused;
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
        <select className="brain-select" aria-labelledby="mode-label" disabled={!ready} value={selectedMode} onChange={event => {
          const choice = MODES.find(option => option.mode === event.target.value);
          if (choice) send(choice.message);
        }}>
          {MODES.map(({ mode: option, label }) => <option key={option} value={option}>{label}</option>)}
        </select>
      </div>
    </div>
    <div className="control-caption"><span>{paused ? 'The simulation is paused. Resume to change experiment settings.' : pending ? <span className="pending-inline" role="status"><i className="spinner"/>{pending.message} Waiting for the next brain frame.</span> : frame ? `${LAYOUTS.find(item => item.layout === layout)?.hint} ${MODES.find(item => item.mode === mode)?.hint}` : status === 'live' ? 'Connected to the brain server. Waiting for the first simulation frame.' : 'Connect a brain server to begin. You can explore the measured anatomy below.'}</span><span className="fixed-synapses">Synaptic weights stay fixed</span></div>
  </section>;
}
