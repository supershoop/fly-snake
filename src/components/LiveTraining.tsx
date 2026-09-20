import type { LiveFrame, LiveStatus, PendingCommand } from '../lib/live';
import { AudienceTraining } from './AudienceTraining';
import { EventStimuli } from './EventStimuli';
import { LearningChart, type LearningRun } from './LearningChart';

type Props = { frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; feedbackUrls: string[]; runs: LearningRun[]; send: (message: object) => void };

export function LiveTraining({ frame, status, paused, pending, feedbackUrls, runs, send }: Props) {
  const supported = frame?.learning.feedback !== undefined && frame?.move !== undefined;
  const canTrain = status === 'live' && frame !== null && supported && !paused && !frame.manual;
  const learning = frame?.policy === 'learning';
  const recent = frame?.learning.scores.slice(-20) ?? [];
  const average = recent.length ? (recent.reduce((sum, score) => sum + score, 0) / recent.length).toFixed(1) : '–';
  const availability = status !== 'live' ? 'Connect the brain server to start learning.'
    : !supported ? 'Restart the brain server to enable live learning.'
    : paused || frame?.manual ? 'Resume normal play to change the learner.'
    : learning ? 'Learning is running. Changes are kept for this server session.'
    : 'Choose a starting point to begin a live-learning run.';

  return <div className="learning-console">
    <div className="learning-column">
      <section className="learning-overview" aria-labelledby="learning-start-title">
        <div>
          <strong id="learning-start-title">START A LEARNING RUN</strong>
          <p>Food and collisions teach the readout automatically. Pick a starting point, then watch the curve below.</p>
        </div>
        <div className="learning-stats" aria-label="Current learning progress">
          <span><b>{learning ? frame?.learning.moves.toLocaleString('en-US') : '—'}</b><small>moves</small></span>
          <span><b>{learning ? frame?.learning.games.toLocaleString('en-US') : '—'}</b><small>games</small></span>
          <span><b>{learning ? average : '—'}</b><small>last-20 avg.</small></span>
        </div>
        <div className="controls learning-actions">
          <button className="learning-primary" disabled={!canTrain} onClick={() => send({ learning: 'pretrained' })} title="Replace the live session with a trainable copy of the trained readout">{pending?.policy === 'learning' ? 'Preparing learner…' : 'Use trained starting point'}</button>
          <button disabled={!canTrain} onClick={() => send({ learning: 'reset' })} title="Replace the live session with a blank readout">{pending?.policy === 'learning' ? 'Preparing learner…' : 'Start from blank'}</button>
        </div>
        <p className="learning-availability" role="status">{availability}</p>
      </section>
      <section className="learning-graph" aria-label="Learning progress">
        <LearningChart runs={runs} current={frame?.wiring ?? 'real'}/>
      </section>
    </div>
    <div className="learning-column">
      <section className="learning-stimuli" aria-label="Automatic event stimuli">
        <EventStimuli frame={frame} status={status} paused={paused} send={send}/>
      </section>
      <AudienceTraining urls={feedbackUrls}/>
    </div>
  </div>;
}
