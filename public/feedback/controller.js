const $ = id => document.getElementById(id);
const canvas = $('board');
const context = canvas.getContext('2d');
let socket, frame = null, pending = null, lastFrameAt = 0, sequence = 0;
const visitor = Math.random().toString(36).slice(2);
const target = () => $('fly').value === 'selected' ? frame?.selected : Number($('fly').value);
const fresh = () => frame && Date.now() - lastFrameAt < 5000 && socket?.readyState === WebSocket.OPEN;

function update() {
  const fly = frame?.flies[target()];
  const reason = !fresh() ? 'Waiting for the live game. Check your connection to the host.'
    : frame.paused ? 'The host paused the game. Feedback will return when play resumes.'
    : frame.manual ? 'The host is applying sensory input. Wait for normal play.'
    : frame.policy !== 'learning' ? 'Waiting for the host to choose Training mode.'
    : !fly?.feedbackEligible ? 'Waiting for this fly to make its next move…' : '';
  $('help').textContent = reason || `Send feedback for fly ${target() + 1}’s displayed move.`;
  $('positive').disabled = $('negative').disabled = Boolean(reason || pending);
  $('connection').textContent = fresh() ? 'Connected to live game' : 'Reconnecting…';
  $('connection').classList.toggle('live', Boolean(fresh()));
  if (!frame) return;
  $('totals').textContent = `${frame.feedback.positive} positive · ${frame.feedback.negative} negative stimuli in this session`;
  const arena = frame.arenas[fly?.arena];
  if (!arena) return;
  const snake = arena.snakes[fly.snake];
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
      $('receipt').textContent = receipt.status === 'applied'
        ? `${receipt.value > 0 ? 'Positive' : 'Negative'} stimulus applied to fly ${receipt.fly + 1} · move ${receipt.move}.`
        : receipt.reason;
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

function send(value) {
  if ($('positive').disabled || !fresh()) return;
  pending = `${visitor}-${++sequence}`;
  socket.send(JSON.stringify({ id: pending, feedback: value, fly: target(), move: frame.move }));
  $('receipt').classList.remove('rejected');
  $('receipt').textContent = 'Sending stimulus…';
  update();
}
$('positive').onclick = () => send(1);
$('negative').onclick = () => send(-1);
$('fly').onchange = update;
setInterval(update, 1000);
connect();
