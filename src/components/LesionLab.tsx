import { useMemo, useState } from 'react';
import type { LiveFrame, NeuronType } from '../lib/live';

/** Relay cells found by path search in the connectome (scripts/lesion_scores.py), not chosen by hand-tuning. */
const PRESETS: { label: string; hint: string; types: string[] }[] = [
  { label: 'Food relays · AOTU', hint: 'LC10 has no direct synapses onto the steering neuron DNa02. The signal crosses 6 cells: AOTU025, AOTU012, AOTU015.', types: ['AOTU025', 'AOTU012', 'AOTU015'] },
  { label: 'Steering · DNa02', hint: 'The descending neuron that turns the fly toward a target.', types: ['DNa02'] },
  { label: 'Turn-away · DNa01', hint: 'Fires on the side opposite a looming threat.', types: ['DNa01'] },
  { label: 'Threat relays · PVLP', hint: 'LC4 reaches the opposite DNa01 through PVLP141 and PVLP137.', types: ['PVLP141', 'PVLP137'] },
  { label: 'Giant fiber · DNp01', hint: 'The escape neuron. LC4 synapses onto it directly.', types: ['DNp01'] },
  { label: 'Object detectors · LC10', hint: 'The cells we stimulate for food, so this removes the food sense itself.', types: ['LC10.*'] },
  { label: 'Looming detectors · LC4', hint: 'The cells we stimulate for side threats.', types: ['LC4'] },
];

export function LesionLab({ frame, types, send }: { frame: LiveFrame | null; types: NeuronType[]; send: (message: object) => void }) {
  const [query, setQuery] = useState('');
  const [everyFly, setEveryFly] = useState(true);
  const active = frame?.flies[frame.selected]?.lesion ?? [];
  const cellCount = useMemo(() => new Map(types.map(([kind, cells]) => [kind, cells])), [types]);
  const matches = useMemo(() => {
    const wanted = query.trim().toLowerCase();
    return wanted ? types.filter(([kind]) => kind.toLowerCase().includes(wanted)).slice(0, 12) : [];
  }, [query, types]);
  const apply = (next: string[]) => send({ lesion: { fly: everyFly ? null : frame?.selected, types: next } });
  const toggle = (patterns: string[]) => apply(patterns.every(p => active.includes(p)) ? active.filter(p => !patterns.includes(p)) : [...new Set([...active, ...patterns])]);
  const silenced = active.reduce((total, pattern) => total + (cellCount.get(pattern) ?? types.filter(([kind]) => new RegExp(`^(?:${pattern})$`).test(kind)).reduce((sum, [, cells]) => sum + cells, 0)), 0);

  return <section className="model-status lesion-lab" aria-label="Lesion lab">
    <div><strong>LESION LAB · SILENCE NEURONS</strong>
      <p>Silenced neurons can never spike; nothing else changes. {active.length ? `${silenced.toLocaleString('en-US')} of ${frame?.totalNeurons.toLocaleString('en-US')} neurons silenced.` : 'The brain is intact.'}</p>
      <div className="controls">{PRESETS.map(preset => <button key={preset.label} title={preset.hint} aria-pressed={preset.types.every(p => active.includes(p))} onClick={() => toggle(preset.types)}>{preset.label}</button>)}
        <button disabled={!active.length} onClick={() => apply([])}>Heal</button></div>
    </div>
    <div><strong>ANY NEURON TYPE</strong>
      <p><label>Search {types.length.toLocaleString('en-US')} annotated types <input value={query} onChange={event => setQuery(event.target.value)} placeholder="e.g. AOTU, MBON, DNg" spellCheck={false}/></label></p>
      <div className="controls">{matches.map(([kind, cells, superclass]) => <button key={kind} aria-pressed={active.includes(kind)} title={superclass} onClick={() => toggle([kind])}>{kind} <small>×{cells}</small></button>)}</div>
      {active.length > 0 && <p>Silenced: {active.map(pattern => <button key={pattern} className="chip" title="Remove" onClick={() => toggle([pattern])}>{pattern.replace('.*', '')} ✕</button>)}</p>}
      <p><label><input type="checkbox" checked={everyFly} onChange={event => setEveryFly(event.target.checked)}/> apply to every fly (off: only the selected fly, so lesioned and intact flies can race)</label></p>
    </div>
  </section>;
}
