import type { LiveFrame, LiveStatus, PendingCommand } from '../lib/live';
import { AudienceTraining } from './AudienceTraining';
import { EventStimuli } from './EventStimuli';

type Props = { frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; feedbackUrls: string[]; send: (message: object) => void };

export function LiveTraining({ frame, status, paused, pending, feedbackUrls, send }: Props) {
  const supported = frame?.learning.feedback !== undefined && frame?.move !== undefined;
  const canTrain = status === 'live' && frame !== null && supported && !paused && !frame.manual;
  const learning = frame?.policy === 'learning';
  const recent = frame?.learning.scores.slice(-20) ?? [];
  const average = recent.length ? (recent.reduce((sum, score) => sum + score, 0) / recent.length).toFixed(1) : '–';

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
    <EventStimuli frame={frame} status={status} paused={paused} send={send}/>
    <AudienceTraining urls={feedbackUrls}/>
  </>;
}
