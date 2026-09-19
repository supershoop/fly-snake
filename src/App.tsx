import { useEffect, useMemo, useState } from 'react';
import { BrainScene } from './components/BrainScene';
import { FlyScene } from './components/FlyScene';
import { Environment } from './components/Environment';
import { Attribution } from './components/Attribution';
import { asset, loadAtlas, type Atlas } from './lib/atlas';
import { useLiveBrain, type Mode } from './lib/live';

const MODES: { mode: Mode; label: string; hint: string }[] = [
  { mode: 'real', label: 'Real wiring + readout', hint: 'Unmodified MaleCNS connectome; a linear readout of its descending neurons picks the move.' },
  { mode: 'hardwired', label: 'Real wiring, no learning', hint: 'Turn toward whichever side’s steering neurons (DNa02, DNa01) fire more. Nothing is trained.' },
  { mode: 'shuffled', label: 'Scrambled wiring', hint: 'Control: same neurons and synapse strengths, random targets, readout retrained.' },
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
  const stimulate = (name: string | null) => { setHeld(name); send({ stimulate: name ? { [name]: 1 } : null }); };
  const steerMax = Math.max(50, ...Object.values(frame?.steer ?? {}));
  return <>
    <header><h1>FLY SNAKE</h1><span>A simulated fruit-fly connectome plays Snake</span><a href="https://github.com/supershoop/fly-snake#readme">About ↗</a></header>
    <main>
      <div className="toolbar">
        <span className="status">{status === 'live' ? (paused ? 'Paused' : 'Live') : status === 'connecting' ? 'Connecting…' : 'Brain server offline'}
          {frame && ` · brain time ${frame.time.toFixed(1)} s · game ${frame.episode} · score ${frame.snake.score} · best ${frame.best}`}</span>
        <div className="controls">
          {MODES.map(({ mode, label, hint }) => <button key={mode} title={hint} aria-pressed={frame?.mode === mode} onClick={() => send({ mode })}>{label}</button>)}
          <button onClick={() => { send({ paused: !paused }); setPaused(!paused); }}>{paused ? 'Resume' : 'Pause'}</button>
        </div>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="workbench">
        <section className="panel environment-panel"><h2>01 / ENVIRONMENT <span>Snake · egocentric senses</span></h2><Environment frame={frame}/>
          <div className="panel-bottom">Food is shown to the fly as a small visual object, walls and body as looming threats</div></section>
        <section className="panel brain-panel"><h2>02 / BRAIN SOMA ATLAS <span>MaleCNS v1.0</span></h2>
          {atlas ? <BrainScene atlas={atlas} frame={activity}/> : <p className="loading" role="status">Loading measured anatomy…</p>}
          <div className="panel-bottom">{frame ? `${frame.activeNeurons.toLocaleString('en-US')} of ${frame.totalNeurons.toLocaleString('en-US')} simulated neurons spiked in the last 100 ms` : `${atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} measured somata`} <a href={asset('data/brain-atlas/NOTICE.md')}>Data notice ↗</a></div>
        </section>
        <section className="panel fly-panel"><h2>03 / BODY <span>Flybody</span></h2><FlyScene/><div className="panel-bottom">Anatomical mesh · no motor simulation <span>Drag to rotate</span></div></section>
      </div>
      <section className="model-status readout" aria-label="Brain output">
        <div><strong>DESCENDING NEURONS (Hz, left / right)</strong>
          {['DNa02', 'DNa01', 'DNp01'].map(kind => <div key={kind} className="steer-row"><span>{kind}{kind === 'DNp01' ? ' · giant fiber' : ' · steering'}</span>
            <meter min={0} max={steerMax} value={frame?.steer[`${kind}_L`] ?? 0}/><meter min={0} max={steerMax} value={frame?.steer[`${kind}_R`] ?? 0}/>
            <span>{frame ? `${frame.steer[`${kind}_L`]} / ${frame.steer[`${kind}_R`]}` : '–'}</span></div>)}
        </div>
        <div><strong>CHOSEN MOVE</strong>
          {ACTIONS.map((label, index) => <div key={label} className="steer-row"><span className={frame?.action === index ? 'chosen' : ''}>{label}</span>
            <meter min={0} max={1} value={frame?.probabilities[index] ?? 0}/><span>{frame ? `${Math.round(frame.probabilities[index] * 100)}%` : '–'}</span></div>)}
        </div>
        <div><strong>STIMULATE BY HAND</strong><p>Hold to override the game’s senses and watch the brain and the move respond.</p>
          <div className="controls">{MANUAL.map(name => <button key={name} aria-pressed={held === name}
            onPointerDown={() => stimulate(name)} onPointerUp={() => stimulate(null)} onPointerLeave={() => held === name && stimulate(null)}>{name}</button>)}</div>
        </div>
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
