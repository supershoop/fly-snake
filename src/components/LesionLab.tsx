import { useMemo, useState } from 'react';
import type { LiveFrame, LiveStatus, NeuronType } from '../lib/live';

/** Relay cells found by path search in the connectome (scripts/lesion_scores.py), not chosen by hand-tuning. */
const PRESETS: { label: string; hint: string; types: string[] }[] = [
  { label: 'Food relays · AOTU', hint: 'LC10 has no direct synapses onto the steering neuron DNa02. The signal crosses 12 cells: AOTU025, AOTU012, AOTU015.', types: ['AOTU025', 'AOTU012', 'AOTU015'] },
  { label: 'Steering · DNa02', hint: 'The descending neuron that turns the fly toward a target.', types: ['DNa02'] },
  { label: 'Turn-away · DNa01', hint: 'Fires on the side opposite a looming threat.', types: ['DNa01'] },
  { label: 'Threat relays · PVLP', hint: 'LC4 reaches the opposite DNa01 through PVLP141 and PVLP137.', types: ['PVLP141', 'PVLP137'] },
  { label: 'Giant fiber · DNp01', hint: 'The escape neuron. LC4 synapses onto it directly.', types: ['DNp01'] },
  { label: 'Object detectors · LC10', hint: 'The cells we stimulate for food, so this removes the food sense itself.', types: ['LC10.*'] },
  { label: 'Looming detectors · LC4', hint: 'The cells we stimulate for side threats.', types: ['LC4'] },
];

/** Rendered only while at least one fly is picked; every action applies to exactly the picked flies. */
export function LesionLab({ frame, status, paused = false, types, picked, send }: { frame: LiveFrame; status: LiveStatus; paused?: boolean; types: NeuronType[]; picked: number[]; send: (message: object) => void }) {
  const [query, setQuery] = useState('');
  const enabled = status === 'live' && !paused;
  const flies = picked.filter(fly => fly < frame.flies.length);
  const lesions = flies.map(fly => frame.flies[fly].lesion);
  const silencedAnywhere = [...new Set(lesions.flat())];
  const inAll = (patterns: string[]) => lesions.length > 0 && lesions.every(list => patterns.every(p => list.includes(p)));
  const matches = useMemo(() => {
    const wanted = query.trim().toLowerCase();
    return wanted ? types.filter(([kind]) => kind.toLowerCase().includes(wanted)) : [];
  }, [query, types]);
  const cellsFor = (pattern: string) => {
    let matcher: RegExp;
    try { matcher = new RegExp(`^(?:${pattern})$`); } catch { return 0; }
    return types.reduce((total, [kind, cells]) => total + (matcher.test(kind) ? cells : 0), 0);
  };
  const setLesion = (index: number, next: string[]) => { if (enabled) send({ lesion: { fly: flies[index], types: next } }); };
  const toggle = (patterns: string[]) => {
    const remove = inAll(patterns);
    flies.forEach((_, i) => setLesion(i, remove ? lesions[i].filter(p => !patterns.includes(p)) : [...new Set([...lesions[i], ...patterns])]));
  };
  const names = flies.length === 1 ? `fly ${flies[0] + 1}` : `flies ${flies.map(fly => fly + 1).join(', ')}`;
  const searchHelp = !types.length ? 'Loading the neuron catalogue…'
    : !query.trim() ? 'Search by type name to find individual neuron groups.'
    : !matches.length ? `No neuron types match “${query.trim()}”.`
    : `${matches.length.toLocaleString('en-US')} matching ${matches.length === 1 ? 'type' : 'types'}${matches.length > 12 ? ' · showing the first 12; refine your search for more' : ''}.`;

  return <section className="model-status lesion-lab" aria-label="Lesion lab">
    <div><strong>SILENCE NEURONS · {names.toUpperCase()}</strong>
      <p>Silenced neurons cannot spike; nothing else changes. Pick more boards to act on several flies at once, and compare them with the intact ones.</p>
      {frame.policy !== 'hardwired' && frame.policy !== 'instinct' && <p className="lesion-note" role="note">In {frame.policy === 'learning' ? 'Training' : 'Trained'} mode the readout listens to all 1,314 descending neurons and works around missing ones, so lesions rarely change how the snake plays. Switch Brain to <strong>Normal</strong> to see the behaviour break.</p>}
      <div className="controls">{PRESETS.map(preset => <button key={preset.label} disabled={!enabled} title={preset.hint} aria-pressed={inAll(preset.types)} onClick={() => toggle(preset.types)}>{preset.label}</button>)}</div>
      <p>{silencedAnywhere.length ? <>Silenced: {silencedAnywhere.map(pattern => <button key={pattern} disabled={!enabled} className="chip" aria-label={`Remove ${pattern} lesion`} title={`Remove ${pattern} lesion from ${names}`}
        onClick={() => flies.forEach((_, i) => setLesion(i, lesions[i].filter(p => p !== pattern)))}>{pattern.replace('.*', '')} <small>×{cellsFor(pattern)}</small> <span aria-hidden="true">×</span></button>)}
        <button disabled={!enabled} className="chip" onClick={() => flies.forEach((_, i) => setLesion(i, []))}>Heal {names}</button></> : `Nothing is silenced in ${names}.`}</p>
    </div>
    <div><strong>STEERING AND ESCAPE NEURONS · LIVE</strong>
      <p>DNa02 turns the fly toward food (about 250 Hz on the food side). The giant fiber DNp01 fires at threats (up to about 380 Hz).</p>
      <div className="lesion-meters">{flies.map(fly => { const steer = frame.flies[fly].steer, snake = frame.arenas[frame.flies[fly].arena]?.snakes[frame.flies[fly].snake]; return <div key={fly} className="steer-row"><span>Fly {fly + 1}{frame.flies[fly].lesion.length ? ' · lesioned' : ''}</span>
        <meter min={0} max={300} value={steer.DNa02_L ?? 0} aria-label={`Fly ${fly + 1} DNa02 left`}/><meter min={0} max={300} value={steer.DNa02_R ?? 0} aria-label={`Fly ${fly + 1} DNa02 right`}/>
        <span>DNa02 {steer.DNa02_L ?? 0} / {steer.DNa02_R ?? 0} · DNp01 {steer.DNp01_L ?? 0} / {steer.DNp01_R ?? 0} Hz · food {snake?.score ?? 0} · games {snake?.games ?? 0}</span></div>; })}</div>
    </div>
    <div><strong>ANY NEURON TYPE</strong>
      <p><label>Search {types.length.toLocaleString('en-US')} annotated types <input disabled={!types.length} value={query} onChange={event => setQuery(event.target.value)} placeholder="e.g. AOTU, MBON, DNg" spellCheck={false} aria-describedby="lesion-search-help" aria-controls="lesion-search-results"/></label></p>
      <p id="lesion-search-help" role="status">{searchHelp}</p>
      <div id="lesion-search-results" className="controls">{matches.slice(0, 12).map(([kind, cells, superclass]) => <button key={kind} disabled={!enabled} aria-pressed={inAll([kind])} title={superclass} onClick={() => toggle([kind])}>{kind} <small>×{cells}</small></button>)}</div>
    </div>
  </section>;
}
