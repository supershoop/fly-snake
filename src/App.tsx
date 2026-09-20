import { useCallback, useEffect, useMemo, useState } from 'react';
import { BrainScene } from './components/BrainScene';
import { FlyScene, type FlyAnimation, type FlyCommand, type FlyDirection } from './components/FlyScene';
import { Environment } from './components/Environment';
import { Attribution } from './components/Attribution';
import { LiveTraining } from './components/LiveTraining';
import { LesionLab } from './components/LesionLab';
import { FlyView } from './components/FlyView';
import { Leaderboard } from './components/Leaderboard';
import type { LearningRun } from './components/LearningChart';
import { Onboarding } from './components/Onboarding';
import { ExperimentControls } from './components/ExperimentControls';
import { Icon } from './components/Icon';
import { asset, loadAtlas, type Atlas } from './lib/atlas';
import { useLiveBrain } from './lib/live';

const INTRO_COPY = {
  solo: {
    title: 'Customize a fly’s wiring. Watch it play Snake.',
    description: 'Pick a fly, tweak its wiring, and see what it does with a game of Snake.',
  },
  versus: {
    title: 'Can you beat a small, puny fruit fly?',
    description: 'You steer the coral snake. The fruit fly gets its simulated brain. Good luck to both of you.',
  },
  swarm: {
    title: 'Sixteen tiny fly brains. Sixteen games of Snake.',
    description: 'Pick a board, change a fly’s wiring, and see whose little brain keeps going.',
  },
  arena: {
    title: 'Eight flies, one board, and a very busy game of Snake.',
    description: 'Pick a fly, tweak its wiring, and watch the tiny rivalry unfold.',
  },
};

export function App() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [error, setError] = useState('');
  const [showOnboarding, setShowOnboarding] = useState(true);
  const paused = false;  // pausing was removed from the interface; components still accept the flag
  const [picked, setPicked] = useState<number[]>([]);
  const [deathSeconds, setDeathSeconds] = useState(0);
  // learning curves of this session, one per wiring, so a scrambled run can be compared with the real one that came before it
  const [runs, setRuns] = useState<Partial<Record<LearningRun['wiring'], number[]>>>({});
  const [brainResetVersion, setBrainResetVersion] = useState(0);
  const { frame, status, send, types, pending, feedbackUrls, vision } = useLiveBrain();
  useEffect(() => {
    const abort = new AbortController();
    void loadAtlas(abort.signal).then(setAtlas).catch(e => { if (!abort.signal.aborted) setError(String(e)); });
    return () => abort.abort();
  }, []);
  const layout = frame?.layout;
  const learningScores = frame?.policy === 'learning' ? frame.learning.scores : null, learningWiring = frame?.wiring;
  useEffect(() => { if (learningScores && learningWiring) setRuns(current => ({ ...current, [learningWiring]: learningScores })); }, [learningScores?.length, learningWiring]);
  useEffect(() => { if (status === 'live' && deathSeconds > 0) send({ deathHold: deathSeconds }); }, [status, deathSeconds, send]);
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
  const intro = INTRO_COPY[displayLayout ?? 'solo'];
  useEffect(() => setPicked([]), [layout]);
  const pick = (fly: number) => { setPicked(current => current.includes(fly) ? current.filter(f => f !== fly) : [...current, fly]); send({ select: fly }); };
  // Brain view marks the silenced cells of every picked fly (or of the shown fly when none is picked).
  const silencedNow = picked.length ? [...new Set(picked.flatMap(fly => frame?.silencedByFly?.[String(fly)] ?? []))] : frame?.silenced ?? [];
  const silencedKey = silencedNow.join(',');
  const silenced = useMemo(() => silencedNow, [silencedKey]);  // stable identity: changes only when a lesion or the pick does
  const activity = useMemo(() => frame ? { time: frame.time, values: frame.values } : null, [frame]);
  const recent = frame?.learning.scores.slice(-20) ?? [];
  const average = recent.length ? (recent.reduce((a, b) => a + b, 0) / recent.length).toFixed(1) : '—';
  // Steering in versus mode is part of continuous play. Keep the board unobscured
  // while the next server frame applies that input; configuration changes still
  // receive the normal immediate pending feedback.
  const visiblePending = pending?.kind === 'human-move' && frame?.layout === 'versus' ? null : pending;
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
  const finishOnboarding = useCallback(() => setShowOnboarding(false), []);

  return <>
    {showOnboarding && <Onboarding onComplete={finishOnboarding}/>}
    <a className="skip-link" href="#experiment">Skip to experiment</a>
    <header className="site-header">
      <a className="brand" href="#" aria-label="snake flies home"><span className="brand-mark"><img className="brand-logo" src="/snakeflies.svg" alt="" /></span><span>snake flies</span></a>
      <span className="header-caption">A tiny brain experiment that plays Snake</span>
      <nav aria-label="Page navigation"><a href="https://github.com/supershoop/fly-snake#readme" target="_blank" rel="noreferrer">About <span aria-hidden="true">↗</span></a></nav>
    </header>
    <main id="experiment">
      <section className="intro" aria-labelledby="page-title">
        <div><p className="eyebrow">MaleCNS v1.0 <span className="intro-slash">/</span> Interactive simulation</p><h1 id="page-title">{intro.title}</h1><p className="intro-description">{intro.description} The board becomes sensory input, then brain activity, then a move.</p></div>
        <div className="session-status"><span className={`status-pill ${status === 'live' && frame && !paused ? 'is-live' : ''}`} role="status"><i/>{connection}</span><span className="mono">{frame ? `${frame.time.toFixed(1)} s brain time · ${frame.flies.length} ${frame.flies.length === 1 ? 'brain' : 'brains'}${frame.thermal?.gpu != null ? ` · GPU ${Math.round(frame.thermal.gpu)} °C` : ''}` : 'Measured anatomy · simulated activity'}</span></div>
      </section>
      <ExperimentControls frame={frame} status={status} paused={paused} pending={visiblePending} send={send}/>
      {visiblePending && <div className="command-toast" role="status" aria-live="polite"><i className="spinner"/><span>{visiblePending.message}<small>Waiting for the next brain frame…</small></span></div>}
      {frame?.thermal?.state === 'cooling' && <p className="thermal-banner is-cooling" role="status">Cooling down. The GPU reached {Math.round(frame.thermal.gpu ?? 0)} °C, so the simulation is holding still until it drops below the resume temperature.</p>}
      {frame?.thermal?.state === 'slow' && <p className="thermal-banner" role="status">Running warm (GPU {Math.round(frame.thermal.gpu ?? 0)} °C). The simulation is leaving short gaps between moves.</p>}
      {error && <p className="error" role="alert">The brain atlas could not load. {error}</p>}
      <div className="workbench">
        <section className="panel environment-panel" aria-label="Snake game">
          <Environment frame={frame} status={status} paused={paused} pending={visiblePending} picked={picked} onPick={pick} onSelect={select => send({ select })} onHuman={human => send({ human })} onResume={resume}/>
          <div className="panel-bottom"><span>Game state <span aria-hidden="true">→</span> sensory neurons <span aria-hidden="true">→</span> brain <span aria-hidden="true">→</span> move</span><span className="live-dot">{frame ? `Move ${frame.move ?? '—'}` : 'Awaiting input'}</span></div>
        </section>
        <section className="panel brain-panel" aria-labelledby="brain-title">
          <div className="panel-heading"><h2 id="brain-title">The Brain</h2><button className="brain-reset" title="Reset to the default XY view" onClick={() => setBrainResetVersion(version => version + 1)}>Reset view</button></div>
          {atlas ? <BrainScene atlas={atlas} frame={activity} silenced={silenced} pathway={vision?.pathway} pathwayRates={frame?.vision?.pathway} resetVersion={brainResetVersion}/> : <p className="loading" role="status">{error ? 'Brain atlas unavailable' : <><i className="spinner"/>Loading measured anatomy…</>}</p>}
          <div className="panel-bottom"><span>{frame ? `${frame.activeNeurons.toLocaleString('en-US')} neurons active · last 100 ms${silenced.length ? ` · ${silenced.length.toLocaleString('en-US')} silenced cells marked` : ''}` : `${atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} measured somata`}</span><a href={asset('data/brain-atlas/NOTICE.md')} target="_blank" rel="noreferrer">Data <span aria-hidden="true">↗</span></a></div>
        </section>
        <section className="panel fly-panel" aria-label="The organism">
          <FlyScene command={flyCommand} onDeathSceneLength={setDeathSeconds}/>
          <div className="body-caption"><em>Drosophila melanogaster</em><span>Also known as the common fruit fly.</span></div>
          <div className="panel-bottom"><span>live fruit fly reaction:</span><span>Drag to rotate · Scroll to zoom</span></div>
        </section>
      </div>
      {frame?.layout === 'versus' && <Leaderboard frame={frame} send={send}/>}
      {frame && (picked.length > 0 || frame.policy === 'learning') && <section className="experiment-lab" id="lab" aria-label="Experiment lab">
        {picked.length > 0 && <details className="lab-disclosure" open><summary><span className="lab-icon"><Icon name="brain" size={21}/></span><span className="disclosure-title">Lesion lab<small>Silence a circuit in the picked {picked.length === 1 ? 'fly' : 'flies'}. Observe what changes.</small></span><span className="disclosure-tag">{picked.length === 1 ? `Fly ${picked[0] + 1}` : `${picked.length} flies`}</span><span className="disclosure-chevron" aria-hidden="true">+</span></summary><LesionLab frame={frame} status={status} paused={paused} pending={pending} types={types} picked={picked} send={send}/></details>}
        {frame.policy === 'learning' && <details className="lab-disclosure" open><summary><span className="lab-icon"><Icon name="sliders" size={21}/></span><span className="disclosure-title">Live learning<small>Shape the readout with reward and punishment.</small></span><span className="disclosure-tag">Learning active · {frame.learning.games} games · mean {average}</span><span className="disclosure-chevron" aria-hidden="true">+</span></summary><div className="model-status readout learning"><LiveTraining frame={frame} status={status} paused={paused} pending={pending} feedbackUrls={feedbackUrls} runs={(['real', 'shuffled'] as const).flatMap(wiring => runs[wiring] ? [{ wiring, scores: runs[wiring]! }] : [])} send={send}/></div></details>}
      </section>}
      <details className="lab-disclosure fly-view-disclosure"><summary><span className="lab-icon"><Icon name="brain" size={21}/></span><span className="disclosure-title">Fly’s-eye view<small>What the fly is shown, mapped onto its own eye, and the option to play through the retina.</small></span><span className="disclosure-tag">{frame?.encoder === 'retina' ? 'Retina encoder on' : 'Optional'}</span><span className="disclosure-chevron" aria-hidden="true">+</span></summary><FlyView frame={frame} vision={vision} send={send}/></details>
      <details className="scientific-scope"><summary>About the simulation, scope & data sources</summary><p><strong>Simulated, never recorded.</strong> A leaky integrate-and-fire model runs on the MaleCNS v1.0 connectome. The game supplies engineered sensory inputs; the descending neurons pick the move. Default modes keep brain synapses fixed. The separate synaptic experiment loads bounded changes to existing steering connections and uses a fixed DNa02/DNa01 readout; this is engineered plasticity, not a measured fly learning rule.</p><p>The atlas shows curated cell-body positions, not neurite branches or synaptic connections. Points keep their native proportions. Optic, central and descending classes are drawn; nerve-cord neurons are simulated but not shown.</p><p>Game threats include collisions and loss of a route to the moving tail. This is engineered spatial preprocessing, not evidence of biological route planning. Simulated firing rates over 100 ms are divided by 100 Hz and clamped to [0, 1] for display. The LIF model follows Shiu et al. 2024 with MaleCNS scaling.</p><p>Dataset creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. <a href="https://male-cns.janelia.org/download/">MaleCNS data and publication</a>, CC BY 4.0. <a href={asset('data/brain-atlas/manifest.json')}>Source, filters and hashes</a>. This is a modified fly-connectome-template; third-party assets retain their own licenses.</p></details>
    </main>
    <Attribution/>
  </>;
}
