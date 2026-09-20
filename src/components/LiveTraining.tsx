import { useEffect, useState } from 'react';
import type { LiveFrame, LiveStatus, PendingCommand } from '../lib/live';

type Props = { frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; send: (message: object) => void };

export function LiveTraining({ frame, status, paused, pending, send }: Props) {
  const [strength, setStrength] = useState(1);
  const [allFlies, setAllFlies] = useState(false);
  const [feedbackPending, setFeedbackPending] = useState<string | null>(null);
  const feedback = frame?.learning.feedback;
  const supported = feedback !== undefined && frame?.move !== undefined;
  const connected = status === 'live' && frame !== null;
  const learning = frame?.policy === 'learning';
  const canTrain = connected && supported && !paused && !frame?.manual;
  const eligible = allFlies ? frame?.flies.some(fly => fly.feedbackEligible) : frame?.flies[frame.selected]?.feedbackEligible;
  const enabled = canTrain && learning && eligible;
  const recent = frame?.learning.scores.slice(-20) ?? [];
  const average = recent.length ? (recent.reduce((sum, score) => sum + score, 0) / recent.length).toFixed(1) : '–';
  const reason = !connected ? 'Connect the brain server to train live.'
    : !supported ? 'Restart the updated brain server to enable stimuli.'
    : paused ? 'Resume the game to send stimuli.'
    : frame?.manual ? 'Release the sensory override to train game moves.'
    : !learning ? 'Choose Retrain existing readout or Start blank to enable stimuli.'
    : !eligible ? 'Waiting for a game move to train…'
    : `Train ${allFlies ? 'all flies’ displayed moves' : `fly ${frame!.selected + 1}’s displayed move`}.`;
  const last = feedback?.last;
  const result = last?.status === 'applied'
    ? `Applied ${last.value > 0 ? '+' : ''}${last.value.toFixed(2)} to ${last.fly === null ? `${last.targets.length} flies` : `fly ${last.fly + 1}`} · move ${last.move}`
    : last?.status === 'rejected' ? last.reason : 'No human stimuli applied in this session.';
  useEffect(() => { if (feedback?.last) setFeedbackPending(null); }, [feedback?.last]);
  const stimulate = (sign: number) => {
    if (enabled && frame) {
      setFeedbackPending(`Sending ${sign > 0 ? 'positive' : 'negative'} feedback…`);
      send({ feedback: sign * strength, fly: allFlies ? null : frame.selected, move: frame.move });
    }
  };

  return <>
    <div><strong>LIVE LEARNING</strong>
      <p>Retrain a copy of the trained readout, or start blank. Food and collisions also provide automatic feedback.</p>
      <div className="controls">
        <button disabled={!canTrain} onClick={() => send({ learning: 'pretrained' })} title="Replace the live session with a trainable copy of the trained readout">{pending?.policy === 'learning' ? 'Preparing learner…' : 'Retrain existing readout'}</button>
        <button disabled={!canTrain} onClick={() => send({ learning: 'reset' })} title="Replace the live session with a blank readout">{pending?.policy === 'learning' ? 'Preparing learner…' : 'Start blank'}</button>
      </div>
      <p>{learning ? `${frame.learning.moves.toLocaleString('en-US')} moves of experience · last-20 average ${average}` : 'Live training is off.'}</p>
      <p>Only the linear readout learns; brain synapses stay fixed. All flies share the readout. Changes last for this server session.</p>
    </div>
    <div className="training-feedback"><strong>POSITIVE / NEGATIVE STIMULI</strong>
      <p>Reward a move to encourage it in similar situations; punish it to discourage it.</p>
      <label htmlFor="stimulus-strength">Strength
        <input id="stimulus-strength" type="range" min="0.25" max="1" step="0.25" value={strength} onChange={event => setStrength(Number(event.target.value))}/>
        <output htmlFor="stimulus-strength">{strength.toFixed(2)}</output>
      </label>
      {frame && frame.flies.length > 1 && <label className="feedback-scope">
        <input type="checkbox" checked={allFlies} onChange={event => setAllFlies(event.target.checked)}/> Apply to every fly
      </label>}
      <p id="feedback-help">{reason}</p>
      <div className="controls" aria-describedby="feedback-help">
        <button className="stimulus-positive" disabled={!enabled} onClick={() => stimulate(1)}>+ Positive stimulus</button>
        <button className="stimulus-negative" disabled={!enabled} onClick={() => stimulate(-1)}>− Negative stimulus</button>
      </div>
      <p className="feedback-result" role="status" aria-live="polite">{feedbackPending ? <><i className="spinner"/>{feedbackPending}</> : result}</p>
      <p>{feedback?.positive ?? 0} positive · {feedback?.negative ?? 0} negative stimuli applied</p>
    </div>
  </>;
}
