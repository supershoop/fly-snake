const $ = id => document.getElementById(id);
const canvas = $('board');
const context = canvas.getContext('2d');
// Board headings, in the order the simulation uses them.
const DIRECTIONS = ['up', 'right', 'down', 'left'];
let socket, frame = null, pending = null, lastFrameAt = 0, sequence = 0;
const visitor = Math.random().toString(36).slice(2);
const target = () => $('fly').value === 'selected' ? frame?.selected : Number($('fly').value);
const fresh = () => frame && Date.now() - lastFrameAt < 5000 && socket?.readyState === WebSocket.OPEN;

function update() {
  const fly = frame?.flies[target()];
  const arena = frame?.arenas[fly?.arena];
  const snake = arena?.snakes[fly?.snake];
  // A snake turns a quarter turn at most, so the direction it came from is not a legal move.
  // Use the decision's own heading: the board has already turned, and the brain validates
  // votes against the facing the fly decided from, not the one it ended up with.
  const facing = fly?.heading ?? snake?.heading;
  const backwards = facing == null ? null : DIRECTIONS[(facing + 2) % 4];
  const reason = !fresh() ? 'Waiting for the live game. Check your connection to the host.'
    : frame.paused ? 'The host paused the game. Teaching will return when play resumes.'
    : frame.manual ? 'The host is applying sensory input. Wait for normal play.'
    : frame.policy !== 'learning' ? 'Waiting for the host to choose Training mode.'
    : !fly?.feedbackEligible ? 'Waiting for this fly to make its next move…' : '';
  $('help').textContent = reason
    || `Press where fly ${target() + 1} should have moved${backwards ? ` — it cannot turn back ${backwards}` : ''}.`;
  for (const direction of DIRECTIONS) {
    const key = $(direction);
    const disabled = Boolean(reason || pending || direction === backwards);
    const blocked = !reason && direction === backwards;
    // Only touch the DOM when a button's state actually changes; re-applying the same
    // disabled/class on every incoming frame is what made the D-pad appear to flash.
    if (key.disabled !== disabled) key.disabled = disabled;
    if (key.classList.contains('blocked') !== blocked) key.classList.toggle('blocked', blocked);
  }
  $('connection').textContent = fresh() ? 'Connected to live game' : 'Reconnecting…';
  $('connection').classList.toggle('live', Boolean(fresh()));
  if (!frame) return;
  const tally = Object.entries(frame.audience?.directions || {}).filter(([, count]) => count)
    .sort((a, b) => b[1] - a[1]).map(([name, count]) => `${name} ${count}`).join(' · ');
  $('totals').textContent = `${frame.audience?.taught || 0} directions taught in this session${tally ? ` · ${tally}` : ''}`;
  const people = frame.audience?.participants || 1;
  $('crowd').textContent = `${people} ${people === 1 ? 'phone' : 'phones'} connected · one vote each per move`;
  $('share').textContent = `${Math.round((frame.audience?.share ?? 1) * 100)}%`;
  if (!arena) return;
  $('score').textContent = `${snake.score} food`;
  const action = ['turned left', 'went straight', 'turned right'][fly.action];
  $('move').textContent = `Move ${frame.move} · Fly ${target() + 1} ${action}${snake.alive ? '' : ' · collision'}`;
  canvas.setAttribute('aria-label', `Fly ${target() + 1}: ${snake.score} food; ${action}${snake.alive ? '' : '; collision'}`);
  context.clearRect(0, 0, canvas.width, canvas.height);
  const cell = canvas.width / arena.size;
  context.strokeStyle = '#202b21';
  context.lineWidth = 1;
  for (let n = 0; n <= arena.size; n++) {
    context.beginPath(); context.moveTo(n * cell, 0); context.lineTo(n * cell, canvas.height); context.stroke();
    context.beginPath(); context.moveTo(0, n * cell); context.lineTo(canvas.width, n * cell); context.stroke();
  }
  for (const [x, y] of arena.foods) {
    context.fillStyle = '#e9ae7c'; context.beginPath();
    context.arc((x + .5) * cell, (y + .5) * cell, cell * .26, 0, Math.PI * 2); context.fill();
  }
  arena.snakes.forEach((item, index) => item.body.forEach(([x, y], segment) => {
    context.fillStyle = !item.alive ? '#7a6153' : index !== fly.snake ? '#75949c' : segment === 0 ? '#ddf4c5' : '#94bc70';
    context.fillRect(x * cell + 2, y * cell + 2, cell - 4, cell - 4);
  }));
}

function connect() {
  const url = new URL('./ws', location.href);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = socket = new WebSocket(url);
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.frame) {
      frame = message.frame; lastFrameAt = Date.now();
      if ($('fly').options.length !== frame.flies.length + 1) {
        const choice = $('fly').value;
        $('fly').replaceChildren(new Option('Host’s selected fly', 'selected'), ...frame.flies.map((_, i) => new Option(`Fly ${i + 1}`, String(i))));
        $('fly').value = choice === 'selected' || Number(choice) < frame.flies.length ? choice : 'selected';
      }
    }
    if (message.receipt && message.id === pending) {
      pending = null;
      const receipt = message.receipt;
      $('receipt').classList.toggle('rejected', receipt.status === 'rejected');
      $('receipt').textContent = receipt.status !== 'applied' ? receipt.reason
        : receipt.direction ? `Taught ${receipt.direction} on move ${receipt.move} at ${Math.round(receipt.weight * 100)}% of the crowd’s influence.`
        : `${receipt.value > 0 ? 'Positive' : 'Negative'} stimulus applied to fly ${receipt.fly + 1} · move ${receipt.move}.`;
    }
    update();
  };
  ws.onclose = () => {
    frame = null;
    if (pending) $('receipt').textContent = 'Connection lost before confirmation. Check the host before sending more feedback.';
    pending = null; update(); setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}

function teach(direction) {
  if ($(direction).disabled || !fresh()) return;
  pending = `${visitor}-${++sequence}`;
  // The direction is absolute; the brain resolves it against the fly's heading on that move.
  socket.send(JSON.stringify({ id: pending, direction, fly: target(), move: frame.move }));
  $('receipt').classList.remove('rejected');
  $('receipt').textContent = `Teaching ${direction}…`;
  update();
}
for (const direction of DIRECTIONS) $(direction).onclick = () => teach(direction);
document.addEventListener('keydown', event => {
  const key = { ArrowUp: 'up', ArrowRight: 'right', ArrowDown: 'down', ArrowLeft: 'left' }[event.key];
  if (key) { event.preventDefault(); teach(key); }
});
$('fly').onchange = update;
setInterval(update, 1000);
connect();
