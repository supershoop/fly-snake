import { useEffect, useMemo, useState } from 'react';
import { BrainScene } from './components/BrainScene';
import { FlyScene } from './components/FlyScene';
import { Environment } from './components/Environment';
import { Attribution } from './components/Attribution';
import { LiveTraining } from './components/LiveTraining';
import { asset, loadAtlas, type Atlas } from './lib/atlas';
import { useLiveBrain, type Layout, type PolicyName } from './lib/live';

const LAYOUTS: { layout: Layout; label: string; hint: string }[] = [
  { layout: 'solo', label: 'One fly', hint: 'One simulated brain, one snake.' },
  { layout: 'swarm', label: '16 flies', hint: 'Sixteen independent brains simulated in one batch, one board each. They share one readout.' },
  { layout: 'versus', label: 'Human vs fly', hint: 'Arrow keys or WASD steer the red snake. The fly sees your body as a looming threat.' },
  { layout: 'arena', label: 'Shared arena', hint: 'Eight flies on one board competing for food.' },
];
const POLICIES: { policy: PolicyName; label: string; hint: string }[] = [
  { policy: 'trained', label: 'Trained readout', hint: 'A linear readout of the descending neurons, fitted offline. Synapses are never changed.' },
  { policy: 'hardwired', label: 'Nothing trained', hint: 'Turn toward whichever side’s steering neurons (DNa02, DNa01) fire more.' },
  { policy: 'learning', label: 'Learn live', hint: 'The readout starts blank and learns from reward while playing: food +1, death -1, closer +0.1.' },
];
const ACTIONS = ['LEFT', 'STRAIGHT', 'RIGHT'];
const MANUAL = ['food_L', 'food_R', 'danger_L', 'danger_R', 'danger_ahead'];

export function App() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [error, setError] = useState('');
  const [paused, setPaused] = useState(false);
  const [held, setHeld] = useState<string | null>(null);
  const { frame, status, send } = useLiveBrain();
  useEffect(() => {
    const abort = new AbortController();
    void loadAtlas(abort.signal).then(setAtlas).catch(e => { if (!abort.signal.aborted) setError(String(e)); });
    return () => abort.abort();
  }, []);
  const activity = useMemo(() => frame ? { time: frame.time, values: frame.values } : null, [frame]);
  const fly = frame?.flies[frame.selected];
  const recent = frame?.learning.scores.slice(-20) ?? [];
  const average = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
  useEffect(() => {
    const keys: Record<string, string> = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right', w: 'up', s: 'down', a: 'left', d: 'right' };
    const press = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLElement && (event.target.closest('input, select, textarea, button') || event.target.isContentEditable)) return;
      const human = keys[event.key]; if (human) { event.preventDefault(); send({ human }); }
    };
    window.addEventListener('keydown', press);
    return () => window.removeEventListener('keydown', press);
  }, [send]);
  const toggleLesion = (pattern: string, everyFly: boolean) => {
    const current = fly?.lesion ?? [];
    const types = current.includes(pattern) ? current.filter(p => p !== pattern) : [...current, pattern];
    send({ lesion: { fly: everyFly ? null : frame?.selected, types } });
  };
  const stimulate = (name: string | null) => { setHeld(name); send({ stimulate: name ? { [name]: 1 } : null }); };
  const steerMax = Math.max(50, ...Object.values(fly?.steer ?? {}));
  return <>
    <header><h1>FLY SNAKE</h1><span>A simulated fruit-fly connectome plays Snake</span><a href="https://github.com/supershoop/fly-snake#readme">About ↗</a></header>
    <main>
      <div className="toolbar">
        <span className="status">{status === 'live' ? (paused ? 'Paused' : 'Live') : status === 'connecting' ? 'Connecting…' : 'Brain server offline'}
          {frame && ` · brain time ${frame.time.toFixed(1)} s · ${frame.flies.length} ${frame.flies.length === 1 ? 'brain' : 'brains'} · ${frame.learning.games} games · last-20 average ${average.toFixed(1)}`}</span>
        <div className="controls">
          {LAYOUTS.map(({ layout, label, hint }) => <button key={layout} title={hint} aria-pressed={frame?.layout === layout} onClick={() => send({ layout })}>{label}</button>)}
          <span className="gap"/>
          {POLICIES.map(({ policy, label, hint }) => <button key={policy} title={hint} aria-pressed={frame?.policy === policy} onClick={() => send({ policy })}>{label}</button>)}
          <button title="Control: same neurons and synapse strengths, random targets" aria-pressed={frame?.wiring === 'shuffled'} onClick={() => send({ wiring: frame?.wiring === 'shuffled' ? 'real' : 'shuffled', policy: 'trained' })}>Scrambled wiring</button>
          <button onClick={() => { send({ paused: !paused }); setPaused(!paused); }}>{paused ? 'Resume' : 'Pause'}</button>
        </div>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="workbench">
        <section className="panel environment-panel"><h2>01 / ENVIRONMENT <span>Snake · egocentric senses</span></h2><Environment frame={frame} onSelect={select => send({ select })}/>
          <div className="panel-bottom">Food drives visual neurons; walls, bodies and routes that cut off escape drive threat neurons</div></section>
        <section className="panel brain-panel"><h2>02 / BRAIN SOMA ATLAS <span>MaleCNS v1.0</span></h2>
          {atlas ? <BrainScene atlas={atlas} frame={activity}/> : <p className="loading" role="status">Loading measured anatomy…</p>}
          <div className="panel-bottom">{frame ? `${frame.activeNeurons.toLocaleString('en-US')} of ${frame.totalNeurons.toLocaleString('en-US')} simulated neurons spiked in the last 100 ms` : `${atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} measured somata`} <a href={asset('data/brain-atlas/NOTICE.md')}>Data notice ↗</a></div>
        </section>
        <section className="panel fly-panel"><h2>03 / BODY <span>Flybody</span></h2><FlyScene/><div className="panel-bottom">Anatomical mesh · no motor simulation <span>Drag to rotate</span></div></section>
      </div>
      <section className="model-status readout" aria-label="Brain output">
        <div><strong>DESCENDING NEURONS (Hz, left / right)</strong>
          {['DNa02', 'DNa01', 'DNp01'].map(kind => <div key={kind} className="steer-row"><span>{kind}{kind === 'DNp01' ? ' · giant fiber' : ' · steering'}</span>
            <meter min={0} max={steerMax} value={fly?.steer[`${kind}_L`] ?? 0}/><meter min={0} max={steerMax} value={fly?.steer[`${kind}_R`] ?? 0}/>
            <span>{fly ? `${fly.steer[`${kind}_L`]} / ${fly.steer[`${kind}_R`]}` : '–'}</span></div>)}
        </div>
        <div><strong>CHOSEN MOVE</strong>
          {ACTIONS.map((label, index) => <div key={label} className="steer-row"><span className={fly?.action === index ? 'chosen' : ''}>{label}</span>
            <meter min={0} max={1} value={fly?.probabilities[index] ?? 0}/><span>{fly ? `${Math.round(fly.probabilities[index] * 100)}%` : '–'}</span></div>)}
        </div>
        <div><strong>STIMULATE BY HAND</strong><p>Hold to override the game’s senses and watch the brain and the move respond.</p>
          <div className="controls">{MANUAL.map(name => <button key={name} aria-pressed={held === name}
            onPointerDown={() => stimulate(name)} onPointerUp={() => stimulate(null)} onPointerLeave={() => held === name && stimulate(null)}>{name}</button>)}</div>
        </div>
      </section>
      <section className="model-status readout" aria-label="Lesions and live learning">
        <div><strong>LESION · SILENCE A NEURON TYPE</strong><p>Silenced neurons can never spike. Click a board to choose which fly the buttons act on.</p>
          <div className="controls">{(frame?.lesionPresets ?? []).map(pattern => <button key={pattern} aria-pressed={fly?.lesion.includes(pattern) ?? false}
            onClick={event => toggleLesion(pattern, event.shiftKey)} title="Shift-click: apply to every fly">{pattern.replace('.*', '')}</button>)}
            <button onClick={() => send({ lesion: { fly: null, types: [] } })}>Heal all</button></div>
        </div>
        <LiveTraining frame={frame} status={status} paused={paused} send={send}/>
      </section>
      <section className="model-status" aria-label="Model provenance">
        <strong>{frame ? 'PREDICTED OUTPUT · SIMULATION' : 'ANATOMY ONLY'}</strong>
        <p>{frame ? 'Leaky integrate-and-fire simulation of the MaleCNS v1.0 connectome (165k traced neurons, connections ≥ 5 synapses, sign from predicted neurotransmitter), after Shiu et al. 2024. Simulated spikes, not recordings from a fly.' : 'No neural model connected. No activity is generated by default.'}</p>
        {frame && <p>Normalization: firing rate over the last 100 ms / 100 Hz, clamped to [0, 1]. Synaptic weights are never changed; only the linear readout is trained.</p>}
      </section>
      <details><summary>Scientific scope</summary><p>The atlas contains curated cell-body positions, not neurite morphology. Points keep native proportions. The brain filter selects optic, central and descending classes; nerve-cord neurons are simulated but not drawn.</p><p>Dataset creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. <a href="https://male-cns.janelia.org/download/">MaleCNS data and publication</a>, CC BY 4.0. <a href={asset('data/brain-atlas/manifest.json')}>Exact source, filters and hashes</a>.</p><p>Built on a modified copy of fly-connectome-template; template code has a custom attribution-required license. Third-party assets retain their own licenses.</p></details>
    </main>
    <Attribution/>
  </>;
}
