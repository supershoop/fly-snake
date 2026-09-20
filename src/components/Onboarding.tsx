import { useEffect, useRef, useState } from 'react';
import './Onboarding.css';

const EXIT_MS = 650;

/** A first-visit film that fades away to reveal the running simulation underneath. */
export function Onboarding({ onComplete }: { onComplete: () => void }) {
  const [leaving, setLeaving] = useState(false);
  const [finalFrame, setFinalFrame] = useState(false);
  const complete = useRef(onComplete);
  const video = useRef<HTMLVideoElement>(null);
  const still = useRef<HTMLCanvasElement>(null);
  complete.current = onComplete;
  const dismiss = () => setLeaving(true);
  const fadeFromFinalFrame = () => {
    const source = video.current, canvas = still.current;
    if (source && canvas && source.videoWidth && source.videoHeight) {
      canvas.width = source.videoWidth;
      canvas.height = source.videoHeight;
      canvas.getContext('2d')?.drawImage(source, 0, 0, canvas.width, canvas.height);
      setFinalFrame(true);
    }
    dismiss();
  };

  useEffect(() => {
    if (!leaving) return;
    const timeout = window.setTimeout(() => complete.current(), EXIT_MS);
    return () => window.clearTimeout(timeout);
  }, [leaving]);

  return <section className={`onboarding ${leaving ? 'is-leaving' : ''}`} aria-label="snake flies introduction">
    <video ref={video} className={`onboarding-video ${finalFrame ? 'is-frozen' : ''}`} autoPlay muted playsInline preload="auto" onEnded={fadeFromFinalFrame} onError={dismiss}>
      <source src="/onboarding.mp4" type="video/mp4"/>
    </video>
    <canvas ref={still} className={`onboarding-still ${finalFrame ? 'is-visible' : ''}`} aria-hidden="true"/>
    <button className="onboarding-skip" type="button" onClick={dismiss}>Skip intro</button>
  </section>;
}
