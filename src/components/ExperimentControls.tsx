import type { Layout, LiveFrame, LiveStatus, PendingCommand, PolicyName } from '../lib/live';
import { Icon } from './Icon';

const LAYOUTS: { layout: Layout; label: string; detail?: string; hint: string }[] = [
  { layout: 'solo', label: 'One fly', hint: 'One simulated brain. One snake. Follow the signal from senses to movement.' },
  { layout: 'swarm', label: 'Swarm', detail: '16', hint: 'Sixteen independent brains, one board each. Select a board to inspect its fly.' },
  { layout: 'versus', label: 'Versus', hint: 'You are the coral snake. Use arrow keys, WASD, or the direction buttons to play.' },
  { layout: 'arena', label: 'Arena', detail: '8', hint: 'Eight flies compete on one board. Choose a fly to inspect its brain and score.' },
];

export function ExperimentControls({ frame, status, paused, pending, onPause, send }: {
  frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; onPause: () => void; send: (message: object) => void;
}) {
  const ready = status === 'live' && !!frame && !paused;
  const layout = pending?.layout ?? frame?.layout;
  const policy = pending?.policy ?? frame?.policy ?? 'trained';
  const wiring = pending?.wiring ?? frame?.wiring;
  return <section className="experiment-controls" aria-label="Experiment settings">
    <div className="control-row">
      <div className="layout-control">
        <span className="eyebrow" id="layout-label">Environment</span>
        <div className="segmented" role="group" aria-labelledby="layout-label">
          {LAYOUTS.map(({ layout: option, label, detail, hint }) => <button key={option} disabled={!ready} title={hint} aria-pressed={layout === option} onClick={() => send({ layout: option })}>
            <Icon name={option}/><span>{label}</span>{detail && <small>{detail}</small>}
          </button>)}
        </div>
      </div>
      <label className="policy-control"><span className="eyebrow">Decision policy</span>
        <select disabled={!ready} value={policy} onChange={event => send({ policy: event.target.value as PolicyName })}>
          <option value="trained">Trained readout</option><option value="hardwired">Nothing trained</option><option value="learning">Learn live</option>
        </select>
      </label>
      <div className="wiring-control"><span className="eyebrow">Connectome</span>
        <button className="wiring-button" disabled={!ready} aria-pressed={wiring === 'shuffled'} title="Toggle original wiring and scrambled targets. The selected decision policy stays the same." onClick={() => send({ wiring: wiring === 'shuffled' ? 'real' : 'shuffled' })}>
          <span className={`switch ${wiring === 'shuffled' ? 'switched' : ''}`} aria-hidden="true"/>{wiring === 'shuffled' ? 'Scrambled wiring' : 'Real wiring'}
        </button>
      </div>
      <button className="pause-button" disabled={status !== 'live'} onClick={onPause}><Icon name={paused || !frame ? 'play' : 'pause'}/>{paused || !frame ? 'Resume' : 'Pause'}</button>
    </div>
    <div className="control-caption"><span>{pending ? <span className="pending-inline" role="status"><i className="spinner"/>{pending.message} Waiting for the next brain frame.</span> : paused ? 'The simulation is paused. Resume to change experiment settings.' : frame ? LAYOUTS.find(item => item.layout === layout)?.hint : status === 'live' ? 'Connected to the brain server. Waiting for the first simulation frame.' : 'Connect a brain server to begin. You can explore the measured anatomy below.'}</span><span className="fixed-synapses">Synaptic weights stay fixed</span></div>
  </section>;
}
