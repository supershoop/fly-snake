import type { LiveFrame, LiveStatus } from '../lib/live';

export function BrainLearning({ frame, status, send }: {
  frame: LiveFrame | null; status: LiveStatus; send: (message: object) => void;
}) {
  const active = frame?.synaptic?.active;
  return <section className="model-status" aria-label="Learning inside the brain" style={{ margin: '12px 0', padding: '14px 18px' }}>
    <strong>{active ? 'Trained brain connections · fixed steering readout' : 'Learning inside the fly brain'}</strong>
    <p>{active
      ? `${active.changedConnections.toLocaleString()} existing synaptic connections adjusted by game rewards. Sensory input enters the fly connectome; DNa02/DNa01 activity controls movement. Saved training is frozen during play.`
      : 'Try a reward-trained version of the connectome. Only existing connections into steering neurons change; the movement readout stays fixed.'}</p>
    <button disabled={status !== 'live' || !frame?.synaptic?.available} onClick={() => send({ synaptic: !active, paused: false })}>
      {active ? 'Restore original brain' : 'Use trained brain connections'}
    </button>
    {!frame?.synaptic?.available && <small> A trained brain checkpoint is not available on this server.</small>}
    <small> Experimental simulated plasticity. Selecting another controller restores the original brain.</small>
    {frame?.synaptic?.error && <p role="alert">Could not load brain checkpoint: {frame.synaptic.error}</p>}
  </section>;
}
