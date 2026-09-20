import { useEffect, useMemo, useState } from 'react';
import { BrainScene } from './components/BrainScene';
import { FlyScene, type FlyAnimation, type FlyCommand, type FlyDirection } from './components/FlyScene';
import { Environment } from './components/Environment';
import { Attribution } from './components/Attribution';
import { LiveTraining } from './components/LiveTraining';
import { LesionLab } from './components/LesionLab';
import { ExperimentControls } from './components/ExperimentControls';
import { Icon } from './components/Icon';
import { asset, loadAtlas, type Atlas } from './lib/atlas';
import { useLiveBrain } from './lib/live';

export function App() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [error, setError] = useState('');
  const paused = false;  // pausing was removed from the interface; components still accept the flag
  const [picked, setPicked] = useState<number[]>([]);
  const { frame, status, send, types, pending } = useLiveBrain();
  useEffect(() => {
    const abort = new AbortController();
    void loadAtlas(abort.signal).then(setAtlas).catch(e => { if (!abort.signal.aborted) setError(String(e)); });
    return () => abort.abort();
  }, []);
  const layout = frame?.layout;
  const fly = frame?.flies[frame.selected];
  const flyCommand = useMemo<FlyCommand | null>(() => {
    if (!frame || !fly) return null;
    const snake = frame.arenas[fly.arena]?.snakes[fly.snake];
    const headings: FlyDirection[] = ['up', 'right', 'down', 'left'];
    const animation: FlyAnimation = fly.reward <= -1
      ? 'pain'
      : fly.reward >= 1
        ? 'win'
        // The server reports a turn relative to the snake. Its rendered heading
        // is the resulting absolute direction, which is the key a player presses.
        : fly.action === 1 || !snake
          ? 'idle'
          : headings[snake.heading];
    return { animation, sequence: frame.time };
  }, [frame?.time, fly?.arena, fly?.snake, fly?.action, fly?.reward, frame?.arenas]);
  const displayLayout = pending?.layout ?? frame?.layout;
  useEffect(() => setPicked([]), [layout]);
  const pick = (fly: number) => { setPicked(current => current.includes(fly) ? current.filter(f => f !== fly) : [...current, fly]); send({ select: fly }); };
  // Brain view marks the silenced cells of every picked fly (or of the shown fly when none is picked).
  const silencedNow = picked.length ? [...new Set(picked.flatMap(fly => frame?.silencedByFly?.[String(fly)] ?? []))] : frame?.silenced ?? [];
  const silencedKey = silencedNow.join(',');
  const silenced = useMemo(() => silencedNow, [silencedKey]);  // stable identity: changes only when a lesion or the pick does
  const activity = useMemo(() => frame ? { time: frame.time, values: frame.values } : null, [frame]);
  const recent = frame?.learning.scores.slice(-20) ?? [];
  const average = recent.length ? (recent.reduce((a, b) => a + b, 0) / recent.length).toFixed(1) : '—';
  useEffect(() => {
    if (frame?.layout !== 'versus' || paused || frame.manual) return;
    const keys: Record<string, string> = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right', w: 'up', s: 'down', a: 'left', d: 'right' };
    const press = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || (event.target instanceof HTMLElement && (event.target.closest('input, select, textarea, summary') || event.target.isContentEditable))) return;
      const human = keys[event.key] ?? keys[event.key.toLowerCase()];
      if (human) { event.preventDefault(); send({ human }); }
    };
    window.addEventListener('keydown', press);
    return () => window.removeEventListener('keydown', press);
  }, [frame?.layout, frame?.manual, paused, send]);
  const resume = () => send({ paused: false, stimulate: null });
  const connection = status === 'live' ? (frame ? paused ? 'Paused' : 'Live simulation' : 'Waiting for simulation') : status === 'connecting' ? 'Connecting' : 'Server offline';

  return <>
    <a className="skip-link" href="#experiment">Skip to experiment</a>
    <header className="site-header">
      <a className="brand" href="#" aria-label="Fly Snake home"><span className="brand-mark"><Icon name="snake" size={23}/></span><span>fly<span className="brand-divider">/</span>snake</span></a>
      <span className="header-caption">A connectome experiment</span>
      <nav aria-label="Page navigation"><a href="#experiment">Workbench</a><a href="https://github.com/supershoop/fly-snake#readme" target="_blank" rel="noreferrer">About <span aria-hidden="true">↗</span></a></nav>
    </header>
    <main id="experiment">
      <section className="intro" aria-labelledby="page-title">
        <div><p className="eyebrow">MaleCNS v1.0 <span className="intro-slash">/</span> Interactive simulation</p><h1 id="page-title">A fly’s wiring. A game of Snake.</h1><p className="intro-description">Follow sensory signals through a simulated fruit-fly connectome, one move at a time.</p></div>
        <div className="session-status"><span className={`status-pill ${status === 'live' && frame && !paused ? 'is-live' : ''}`} role="status"><i/>{connection}</span><span className="mono">{frame ? `${frame.time.toFixed(1)} s brain time · ${frame.flies.length} ${frame.flies.length === 1 ? 'brain' : 'brains'}` : 'Measured anatomy · simulated activity'}</span></div>
      </section>
      <ExperimentControls frame={frame} status={status} paused={paused} pending={pending} send={send}/>
      {pending && <div className="command-toast" role="status" aria-live="polite"><i className="spinner"/><span>{pending.message}<small>Waiting for the next brain frame…</small></span></div>}
      {error && <p className="error" role="alert">The brain atlas could not load. {error}</p>}
      <div className="workbench">
        <section className="panel environment-panel" aria-labelledby="environment-title">
          <div className="panel-heading"><h2 id="environment-title"><span className="panel-number">01</span>The environment</h2><span className="panel-meta">{displayLayout === 'versus' ? 'Human vs fly' : displayLayout === 'swarm' ? '16 independent boards' : displayLayout === 'arena' ? '8 flies · one board' : 'Snake'}</span></div>
          <Environment frame={frame} status={status} paused={paused} pending={pending} picked={picked} onPick={pick} onSelect={select => send({ select })} onHuman={human => send({ human })} onResume={resume}/>
          <div className="panel-bottom"><span>Game state <span aria-hidden="true">→</span> sensory neurons <span aria-hidden="true">→</span> brain <span aria-hidden="true">→</span> move</span><span className="live-dot">{frame ? `Move ${frame.move ?? '—'}` : 'Awaiting input'}</span></div>
        </section>
        <section className="panel brain-panel" aria-labelledby="brain-title">
          <div className="panel-heading"><h2 id="brain-title"><span className="panel-number">02</span>Brain</h2></div>
          {atlas ? <BrainScene atlas={atlas} frame={activity} silenced={silenced}/> : <p className="loading" role="status">{error ? 'Brain atlas unavailable' : <><i className="spinner"/>Loading measured anatomy…</>}</p>}
          <div className="panel-bottom"><span>{frame ? `${frame.activeNeurons.toLocaleString('en-US')} neurons active · last 100 ms${silenced.length ? ` · ${silenced.length.toLocaleString('en-US')} silenced cells marked` : ''}` : `${atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} measured somata`}</span><a href={asset('data/brain-atlas/NOTICE.md')} target="_blank" rel="noreferrer">Data <span aria-hidden="true">↗</span></a></div>
        </section>
        <section className="panel fly-panel" aria-labelledby="body-title">
          <div className="panel-heading"><h2 id="body-title"><span className="panel-number">03</span>The organism</h2><span className="panel-meta">Drosophila</span></div>
          <FlyScene command={flyCommand}/>
          <div className="body-caption"><em>Drosophila melanogaster</em><span>Rigged motor display<br/>Live Snake controls</span></div>
          <div className="panel-bottom"><span>Live input/reward animation</span><span>Drag to rotate</span></div>
        </section>
      </div>
      {frame && (picked.length > 0 || frame.policy === 'learning') && <section className="experiment-lab" id="lab" aria-label="Experiment lab">
        {picked.length > 0 && <details className="lab-disclosure" open><summary><span className="lab-icon"><Icon name="brain" size={21}/></span><span className="disclosure-title">Lesion lab<small>Silence a circuit in the picked {picked.length === 1 ? 'fly' : 'flies'}. Observe what changes.</small></span><span className="disclosure-tag">{picked.length === 1 ? `Fly ${picked[0] + 1}` : `${picked.length} flies`}</span><span className="disclosure-chevron" aria-hidden="true">+</span></summary><LesionLab frame={frame} status={status} paused={paused} pending={pending} types={types} picked={picked} send={send}/></details>}
        {frame.policy === 'learning' && <details className="lab-disclosure" open><summary><span className="lab-icon"><Icon name="sliders" size={21}/></span><span className="disclosure-title">Live learning<small>Shape the readout with reward and punishment.</small></span><span className="disclosure-tag">Learning active · {frame.learning.games} games · mean {average}</span><span className="disclosure-chevron" aria-hidden="true">+</span></summary><div className="model-status readout learning"><LiveTraining frame={frame} status={status} paused={paused} pending={pending} send={send}/></div></details>}
      </section>}
      <details className="scientific-scope"><summary>About the simulation, scope & data sources</summary><p><strong>Simulated, never recorded.</strong> A leaky integrate-and-fire model runs on the MaleCNS v1.0 connectome. The game supplies engineered sensory inputs; the descending neurons pick the move. Brain synapses stay fixed; only a readout is ever trained.</p><p>The atlas shows curated cell-body positions, not neurite branches or synaptic connections. Points keep their native proportions. Optic, central and descending classes are drawn; nerve-cord neurons are simulated but not shown.</p><p>Game threats include collisions and loss of a route to the moving tail. This is engineered spatial preprocessing, not evidence of biological route planning. Simulated firing rates over 100 ms are divided by 100 Hz and clamped to [0, 1] for display. The LIF model follows Shiu et al. 2024 with MaleCNS scaling.</p><p>Dataset creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. <a href="https://male-cns.janelia.org/download/">MaleCNS data and publication</a>, CC BY 4.0. <a href={asset('data/brain-atlas/manifest.json')}>Source, filters and hashes</a>. This is a modified fly-connectome-template; third-party assets retain their own licenses.</p></details>
    </main>
    <Attribution/>
  </>;
}
