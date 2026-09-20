import { useEffect, useMemo, useState } from 'react';
import type { LiveFrame, LiveStatus, NeuronType, PendingCommand } from '../lib/live';

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

export function LesionLab({ frame, status, paused = false, pending, types, send }: { frame: LiveFrame | null; status: LiveStatus; paused?: boolean; pending: PendingCommand | null; types: NeuronType[]; send: (message: object) => void }) {
  const [query, setQuery] = useState('');
  const [everyFly, setEveryFly] = useState(true);
  const [optimisticActive, setOptimisticActive] = useState<string[] | null>(null);
  const connected = status === 'live' && frame !== null;
  const enabled = connected && !paused;
  const serverActive = frame?.flies[frame.selected]?.lesion ?? [];
  const serverActiveKey = serverActive.join('\u0000');
  useEffect(() => {
    if (optimisticActive && optimisticActive.length === serverActive.length && optimisticActive.every(pattern => serverActive.includes(pattern))) setOptimisticActive(null);
  }, [optimisticActive, serverActive, serverActiveKey]);
  const active = optimisticActive ?? serverActive;
  const matches = useMemo(() => {
    const wanted = query.trim().toLowerCase();
    return wanted ? types.filter(([kind]) => kind.toLowerCase().includes(wanted)) : [];
  }, [query, types]);
  const apply = (next: string[]) => {
    if (enabled && frame) {
      setOptimisticActive(next);
      send({ lesion: { fly: everyFly ? null : frame.selected, types: next } });
    }
  };
  const toggle = (patterns: string[]) => apply(patterns.every(p => active.includes(p)) ? active.filter(p => !patterns.includes(p)) : [...new Set([...active, ...patterns])]);
  const patterns = active.flatMap(pattern => {
    try { return [new RegExp(`^(?:${pattern})$`)]; } catch { return []; }
  });
  const silenced = types.reduce((total, [kind, cells]) => total + (patterns.some(pattern => pattern.test(kind)) ? cells : 0), 0);
  const canHeal = enabled && (everyFly ? frame.flies.some(fly => fly.lesion.length > 0) : active.length > 0);
  const searchHelp = !connected ? 'Connect the brain server to explore neuron types.'
    : !types.length ? 'Loading the neuron catalogue…'
    : !query.trim() ? 'Search by type name to find individual neuron groups.'
    : !matches.length ? `No neuron types match “${query.trim()}”.`
    : `${matches.length.toLocaleString('en-US')} matching ${matches.length === 1 ? 'type' : 'types'}${matches.length > 12 ? ' · showing the first 12; refine your search for more' : ''}.`;

  return <section className="model-status lesion-lab" aria-label="Lesion lab">
    <div><strong>LESION LAB · SILENCE NEURONS</strong>
      <p>Silenced neurons cannot spike. {!connected ? 'Connect the brain server to run a lesion experiment.' : paused ? 'Resume the simulation to change lesions.' : active.length ? `${types.length ? silenced.toLocaleString('en-US') + ' neurons' : active.length + ' type patterns'} silenced in fly ${frame!.selected + 1}.` : 'The selected brain is intact.'}</p>
      {optimisticActive && pending?.message === 'Applying the lesion…' && <p className="pending-text" role="status"><i className="spinner"/>Applying lesion to the simulation…</p>}
      <div className="controls">{PRESETS.map(preset => <button key={preset.label} disabled={!enabled} title={preset.hint} aria-pressed={preset.types.every(p => active.includes(p))} onClick={() => toggle(preset.types)}>{preset.label}</button>)}
        <button disabled={!canHeal} onClick={() => apply([])}>{everyFly ? 'Heal all flies' : 'Heal selected fly'}</button></div>
    </div>
    <div><strong>ANY NEURON TYPE</strong>
      <p><label>Search {types.length.toLocaleString('en-US')} annotated types <input disabled={!connected || !types.length} value={query} onChange={event => setQuery(event.target.value)} placeholder="e.g. AOTU, MBON, DNg" spellCheck={false} aria-describedby="lesion-search-help" aria-controls="lesion-search-results"/></label></p>
      <p id="lesion-search-help" role="status">{searchHelp}</p>
      <div id="lesion-search-results" className="controls">{matches.slice(0, 12).map(([kind, cells, superclass]) => <button key={kind} disabled={!enabled} aria-pressed={active.includes(kind)} title={superclass} onClick={() => toggle([kind])}>{kind} <small>×{cells}</small></button>)}</div>
      {active.length > 0 && <p>Silenced: {active.map(pattern => <button key={pattern} disabled={!enabled} className="chip" aria-label={`Remove ${pattern} lesion`} title={`Remove ${pattern} lesion`} onClick={() => toggle([pattern])}>{pattern.replace('.*', '')} <span aria-hidden="true">×</span></button>)}</p>}
      <p><label><input type="checkbox" disabled={!connected} checked={everyFly} onChange={event => setEveryFly(event.target.checked)}/> Apply to every fly</label></p>
      <p>{everyFly ? 'Changes replace every fly’s lesions with the selection shown here.' : 'Changes affect only the selected fly. Compare it with intact flies.'}</p>
    </div>
  </section>;
}
