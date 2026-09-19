/** Replace this component with your game, video, sensor feed or Gymnasium frontend. */
export function Environment({time}:{time:number}) {
  const x = 160 + Math.sin(time*.7)*100;
  return <div className="environment">
    <svg viewBox="0 0 320 320" role="img" aria-label="Example visual stimulus: a moving spot on a grid">
      <defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#242a30" strokeWidth=".5"/></pattern></defs>
      <rect width="320" height="320" fill="url(#grid)"/>
      <circle cx={x} cy="160" r="18" fill="#d5dce3"/>
      <path d="M160 150v20m-10-10h20" stroke="#566170"/>
    </svg>
    <p>Example visual stimulus</p><span>Replace with your environment. No model is driven by this spot.</span>
  </div>;
}
