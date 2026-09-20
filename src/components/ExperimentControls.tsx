import type { Layout, LiveFrame, LiveStatus } from '../lib/live';
import { Icon } from './Icon';

const LAYOUTS: { layout: Layout; label: string; detail?: string; hint: string }[] = [
  { layout: 'solo', label: 'One fly', hint: 'One simulated brain. One snake. Follow the signal from senses to movement.' },
  { layout: 'swarm', label: 'Swarm', detail: '16', hint: 'Sixteen independent brains, one board each. Select a board to inspect its fly.' },
  { layout: 'versus', label: 'Versus', hint: 'You are the coral snake. Use arrow keys, WASD, or the direction buttons to play.' },
  { layout: 'arena', label: 'Arena', detail: '8', hint: 'Eight flies compete on one board. Choose a fly to inspect its brain and score.' },
];

type Mode = 'trained' | 'normal' | 'scrambled' | 'training';
const MODES: { mode: Mode; label: string; hint: string; message: object }[] = [
  { mode: 'trained', label: 'Trained', hint: 'Real wiring. A linear readout of the descending neurons, fitted offline, picks the move.', message: { wiring: 'real', policy: 'trained' } },
  { mode: 'normal', label: 'Normal', hint: 'Real wiring, nothing trained. Steering neurons (DNa02, DNa01) pull the snake toward food; the giant fiber (DNp01) vetoes turns into a threat and triggers dodges.', message: { wiring: 'real', policy: 'instinct' } },
  { mode: 'scrambled', label: 'Scrambled', hint: 'Control. Same neurons and synapse strengths, random targets, nothing trained.', message: { wiring: 'shuffled', policy: 'instinct' } },
  { mode: 'training', label: 'Training', hint: 'Real wiring. The readout learns while playing, from reward alone.', message: { wiring: 'real', policy: 'learning' } },
];
const modeOf = (frame: LiveFrame): Mode => frame.wiring === 'shuffled' ? 'scrambled' : frame.policy === 'hardwired' || frame.policy === 'instinct' ? 'normal' : frame.policy === 'learning' ? 'training' : 'trained';

export function ExperimentControls({ frame, status, paused, send }: {
  frame: LiveFrame | null; status: LiveStatus; paused: boolean; send: (message: object) => void;
}) {
  const ready = status === 'live' && !!frame && !paused;
  return <section className="experiment-controls" aria-label="Experiment settings">
    <div className="control-row">
      <div className="layout-control">
        <span className="eyebrow" id="layout-label">Environment</span>
        <div className="segmented" role="group" aria-labelledby="layout-label">
          {LAYOUTS.map(({ layout, label, detail, hint }) => <button key={layout} disabled={!ready} title={hint} aria-pressed={frame?.layout === layout} onClick={() => send({ layout })}>
            <Icon name={layout}/><span>{label}</span>{detail && <small>{detail}</small>}
          </button>)}
        </div>
      </div>
      <div className="layout-control">
        <span className="eyebrow" id="mode-label">Brain</span>
        <div className="segmented" role="group" aria-labelledby="mode-label">
          {MODES.map(({ mode, label, hint, message }) => <button key={mode} disabled={!ready} title={hint} aria-pressed={frame ? modeOf(frame) === mode : false} onClick={() => send(message)}><span>{label}</span></button>)}
        </div>
      </div>
    </div>
    <div className="control-caption"><span>{paused ? 'The simulation is paused. Resume to change experiment settings.' : frame ? `${LAYOUTS.find(item => item.layout === frame.layout)?.hint} ${MODES.find(item => item.mode === modeOf(frame))?.hint}` : status === 'live' ? 'Connected to the brain server. Waiting for the first simulation frame.' : 'Connect a brain server to begin. You can explore the measured anatomy below.'}</span><span className="fixed-synapses">Synaptic weights stay fixed</span></div>
  </section>;
}
