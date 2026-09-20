import { useEffect, useState } from 'react';
import type { EventStimulusSettings, LiveFrame, LiveStatus, StimulusChoice } from '../lib/live';
import './EventStimuli.css';

const EVENTS = [
  { key: 'food', label: 'When it gets food', description: 'Each time the snake eats.' },
  { key: 'death', label: 'When it dies', description: 'A collision or a starvation timeout.' },
] as const;
const CHOICES: { value: StimulusChoice; label: string; detail: string }[] = [
  { value: 'positive', label: 'Positive', detail: 'Sugar taste' },
  { value: 'negative', label: 'Negative', detail: 'Heat cue' },
  { value: 'none', label: 'None', detail: 'No added stimulus' },
];

export function EventStimuli({ frame, status, paused, send }: {
  frame: LiveFrame | null; status: LiveStatus; paused: boolean; send: (message: object) => void;
}) {
  const [waiting, setWaiting] = useState<Partial<EventStimulusSettings>>({});
  const settings = frame?.eventStimuli;
  const ready = status === 'live' && !!settings && !paused && !frame?.manual;
  useEffect(() => {
    setWaiting(current => {
      if (status !== 'live' || frame?.eventStimulusError) return {};
      return Object.fromEntries(Object.entries(current).filter(([key, value]) => settings?.[key as keyof EventStimulusSettings] !== value));
    });
  }, [settings?.food, settings?.death, frame?.eventStimulusError, status]);
  const choose = (key: keyof EventStimulusSettings, value: StimulusChoice) => {
    if (!ready) return;
    setWaiting(current => ({ ...current, [key]: value }));
    send({ eventStimuli: { [key]: value } });
  };
  const message = status !== 'live' ? 'Connect the brain server to choose stimuli.'
    : !settings ? 'Restart the updated brain server to choose event stimuli.'
    : paused || frame?.manual ? 'Resume normal play to change stimuli.'
    : frame?.eventStimulusError ? frame.eventStimulusError
    : Object.keys(waiting).length ? 'Applying your choices on the next brain move…'
    : 'Choices applied to every fly · saved for this server session.';

  return <section className="event-stimuli" aria-label="Stimuli from game events">
    <strong>STIMULI FROM GAME EVENTS</strong>
    <p>Choose the sensory stimulus the fly receives automatically when something happens.</p>
    <div className="event-stimuli-rows">
      {EVENTS.map(({ key, label, description }) => <fieldset key={key} disabled={!ready || waiting[key] !== undefined}>
        <legend>{label}</legend>
        <p>{description}</p>
        <div className="event-stimuli-options">
          {CHOICES.map(({ value, label: option, detail }) => <label key={value} className={`event-stimulus-option ${value}`}>
            <input type="radio" name={`event-stimulus-${key}`} value={value}
              checked={(waiting[key] ?? settings?.[key]) === value} onChange={() => choose(key, value)}/>
            <span><b>{option}</b><small>{detail}</small></span>
          </label>)}
        </div>
      </fieldset>)}
    </div>
    <p className="event-stimuli-status" role="status" aria-live="polite">{message}</p>
  </section>;
}
