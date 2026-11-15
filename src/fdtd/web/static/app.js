// Flow: choose a view -> parameters -> split view (3D room + 2D slice).
import { RoomView } from './viewer3d.js';
import { SliceView, divergingMap, sequentialMap, gradientCss } from './slice2d.js';
import { SweepChart, describeMode } from './chart.js';

const $ = (id) => document.getElementById(id);
const AXES = ['w', 'h', 'd'];
const AXIS_NAMES = { w: 'width', h: 'height', d: 'depth' };

const state = {
  options: null,
  mode: null,        // 'slowmotion' | 'sweep'
  info: null,        // session info from the server
  axis: 'h',
  indices: { w: 0, h: 0, d: 0 },
  // slow motion
  playing: false,
  busy: false,
  scale: 1e-9,
  // sweep
  selected: -1,
  pollTimer: null,
  sweepStatus: null,
  // latest request wins: one counter for the slice position, one for the field data
  geometryToken: 0,
  dataToken: 0,
};

// ---------- API ----------

async function api(path, { method = 'GET', body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const isJson = (response.headers.get('Content-Type') || '').includes('application/json');
  if (!response.ok) {
    const payload = isJson ? await response.json() : { error: response.statusText };
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  if (isJson) return response.json();
  const buffer = await response.arrayBuffer();
  const h = (key) => response.headers.get(`X-${key}`);
  const dtype = h('Dtype');
  return {
    data: dtype === 'uint8' ? new Uint8Array(buffer) : new Float32Array(buffer),
    rows: Number(h('Rows')),
    cols: Number(h('Cols')),
    horizontal: h('Horizontal'),
    vertical: h('Vertical'),
    time: h('Time') !== null ? Number(h('Time')) : null,
    iteration: h('Iteration') !== null ? Number(h('Iteration')) : null,
    min: h('Min') !== null ? Number(h('Min')) : null,
    max: h('Max') !== null ? Number(h('Max')) : null,
  };
}

function showError(error) {
  console.error(error);
  $('error-text').textContent = error.message || String(error);
  $('error').hidden = false;
}
$('error-close').addEventListener('click', () => { $('error').hidden = true; });

// ---------- navigation ----------

function showScreen(name) {
  for (const screen of ['mode', 'params', 'viewer']) {
    $(`screen-${screen}`).hidden = screen !== name;
  }
  for (const crumb of document.querySelectorAll('.crumb')) {
    const target = crumb.dataset.goto;
    crumb.classList.toggle('active', target === name);
    crumb.disabled = (target === 'params' && !state.mode) ||
      (target === 'viewer' && (!state.info || state.info.mode !== state.mode));
  }
  if (name !== 'viewer') leaveViewer();
  if (name === 'viewer') {
    roomView?.resize();
    sliceView?.layout();
    chart?.draw();
  }
}

for (const button of document.querySelectorAll('[data-goto]')) {
  button.addEventListener('click', () => showScreen(button.dataset.goto));
}

function leaveViewer() {
  setPlaying(false);
}

// ---------- 1: view ----------

for (const card of document.querySelectorAll('.mode-card')) {
  card.addEventListener('click', () => {
    state.mode = card.dataset.mode;
    $('params-title').textContent = state.mode === 'sweep' ? 'Full sweep parameters' : 'Slow motion parameters';
    for (const section of document.querySelectorAll('[data-for]')) {
      section.hidden = section.dataset.for !== state.mode;
    }
    updateEstimate();
    showScreen('params');
  });
}

// ---------- 2: parameters ----------

const form = $('params-form');

function fillSelect(select, items) {
  select.innerHTML = '';
  for (const { value, label } of items) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }
}

function currentScene() {
  return state.options.scenes.find((s) => s.key === form.scene.value);
}

function applySceneDefaults() {
  const scene = currentScene();
  const [w, h, d] = scene.size;
  $('scene-description').textContent =
    `${scene.description ? `${scene.description} ` : ''}${w} × ${h} × ${d} m (w × h × d), ${(w * h * d).toFixed(1)} m³.`;
  ['source_w', 'source_h', 'source_d'].forEach((name, i) => {
    form[name].value = scene.default_source[i];
    form[name].max = scene.size[i];
  });
  updateEstimate();
}

function bandCount(low, high, fraction) {
  if (!(low > 0 && high >= low && fraction > 0)) return 0;
  const lowerIt = fraction * -Math.log2(low / 1000);
  const upperIt = fraction * -Math.log2(high / 1000);
  return Math.max(0, Math.ceil(lowerIt) - Math.floor(upperIt) + 1);
}

function formatCount(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return `${Math.round(n)}`;
}

function formatBytes(n) {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MiB`;
  return `${(n / 1024).toFixed(0)} KiB`;
}

function formatDuration(seconds) {
  if (seconds === null || !Number.isFinite(seconds)) return '–';
  if (seconds < 60) return `${seconds.toFixed(0)} s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

function updateEstimate() {
  if (!state.options) return;
  const scene = currentScene();
  const c = state.options.c_air;
  const fmax = Number(form.max_frequency.value);
  const oversampling = Number(form.oversampling.value);
  const scheme = state.options.schemes.find((s) => s.key === form.scheme.value);
  const box = $('estimate');
  if (!(fmax > 0 && oversampling > 0)) {
    box.innerHTML = '';
    return;
  }
  const dx = c / (fmax * oversampling);
  const dt = (dx * scheme.lambda) / c;
  const shape = scene.size.map((s) => Math.max(1, Math.round(s / dx)));
  const cells = shape[0] * shape[1] * shape[2];
  const stats = [
    ['Cell size', `${(dx * 100).toFixed(1)} cm`],
    ['Grid', `${shape.join(' × ')}`],
    ['Cells', formatCount(cells)],
    ['Time step', `${(dt * 1e6).toFixed(1)} µs`],
  ];
  const warnings = [];
  if (cells > state.options.max_cells) warnings.push(`Too many cells (limit ${formatCount(state.options.max_cells)}).`);
  if (state.mode === 'sweep') {
    const bands = bandCount(Number(form.sweep_min.value), Number(form.sweep_max.value), Number(form.bands.value));
    const steps = Math.max(1, Math.floor(Number(form.sweep_time.value) / dt));
    stats.push(['Frequencies', `${bands}`]);
    stats.push(['Steps / frequency', formatCount(steps)]);
    stats.push(['Cell updates', formatCount(bands * steps * cells)]);
    stats.push(['Stored fields', formatBytes(bands * cells * 4)]);
    if (Number(form.sweep_max.value) > fmax) warnings.push('The sweep goes above the grid max frequency.');
  } else {
    stats.push(['Simulated per step', `${(dt * 1000).toFixed(3)} ms`]);
    if (Number(form.signal_frequency.value) > fmax) warnings.push('The signal frequency is above the grid max frequency; expect dispersion.');
  }
  box.innerHTML = stats.map(([label, value]) =>
    `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${value}</span></div>`).join('') +
    warnings.map((w) => `<div class="stat-warning">${w}</div>`).join('');
}

form.addEventListener('input', updateEstimate);
form.scene.addEventListener('change', applySceneDefaults);

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(form).entries());
  const body = {
    mode: state.mode,
    scene: values.scene,
    max_frequency: Number(values.max_frequency),
    oversampling: Number(values.oversampling),
    scheme: values.scheme,
    precision: values.precision,
    source_type: values.source_type,
    source: [values.source_w, values.source_h, values.source_d].map(Number),
  };
  if (state.mode === 'slowmotion') {
    Object.assign(body, { generator: values.generator, signal_frequency: Number(values.signal_frequency) });
  } else {
    Object.assign(body, {
      sweep_min: Number(values.sweep_min),
      sweep_max: Number(values.sweep_max),
      bands: Number(values.bands),
      sweep_time: Number(values.sweep_time),
    });
  }
  const button = $('start-button');
  button.disabled = true;
  button.textContent = 'Building grid…';
  $('error').hidden = true;
  try {
    const info = await api('/api/session', { method: 'POST', body });
    await openViewer(info);
  } catch (error) {
    // the server drops the previous session before building a new one
    stopPolling();
    state.info = null;
    showScreen('params');
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = 'Build & start';
  }
});

// ---------- 3: viewer ----------

let roomView = null;
let sliceView = null;
let chart = null;

function ensureViews() {
  if (roomView) return;
  roomView = new RoomView($('view3d'), $('hover-label'));
  sliceView = new SliceView({
    box: $('slice-box'),
    canvas: $('slice-canvas'),
    tooltip: $('slice-tooltip'),
    axisX: $('axis-x'),
    axisY: $('axis-y'),
  });
  sliceView.onImageChange = () => roomView.updateTexture();
  chart = new SweepChart($('chart-canvas'), $('chart-tooltip'), (index) => {
    $('follow-latest').checked = false;
    selectFrequency(index);
  });
  for (const box of document.querySelectorAll('[data-layer]')) {
    box.addEventListener('change', () => roomView.setLayerVisible(box.dataset.layer, box.checked));
  }
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    sliceView.map = state.info?.mode === 'sweep' ? sequentialMap() : divergingMap();
    sliceView.paintImage();
    sliceView.draw();
    updateColorbar();
    chart.draw();
  });
}

async function openViewer(info) {
  stopPolling();
  setPlaying(false);
  state.info = info;
  ensureViews();
  const isSweep = info.mode === 'sweep';
  const { grid, scene, parameters } = info;

  $('viewer-title').textContent = `${scene.name} · ${isSweep ? 'Full sweep' : 'Slow motion'}`;
  const signal = isSweep
    ? `${info.sweep.frequencies.length} frequencies, ${info.sweep.time_per_frequency} s each`
    : `${info.generator} @ ${parameters.signal_frequency} Hz`;
  $('viewer-sub').textContent =
    `${grid.shape.join(' × ')} cells (${formatCount(grid.cells)}), dx ${(grid.dx * 100).toFixed(1)} cm, ` +
    `dt ${(grid.dt * 1e6).toFixed(1)} µs, ${parameters.scheme.toUpperCase()} ${parameters.precision}, ` +
    `${parameters.source_type} source · ${signal}`;
  $('device').textContent = info.device;

  $('slowmotion-controls').hidden = isSweep;
  $('sweep-controls').hidden = !isSweep;
  $('sweep-panel').hidden = !isSweep;
  $('hold-scale-label').hidden = isSweep;
  $('slice-title').textContent = isSweep ? 'Sound level (Leq) in the slice' : 'Pressure in the slice';
  $('cb-unit').textContent = isSweep ? 'dB' : 'rel. Pa';

  // slice through the source by default
  AXES.forEach((axis, i) => { state.indices[axis] = info.source.cell[i]; });
  sliceView.data = null;
  roomView.setScene(info, sliceView.image);
  showScreen('viewer');
  setAxis(state.axis);

  if (isSweep) {
    state.selected = -1;
    state.sweepStatus = null;
    $('follow-latest').checked = true;
    $('freq-slider').max = 0;
    $('freq-slider').value = 0;
    $('freq-value').textContent = 'waiting for the first frequency…';
    $('mode-hint').textContent = '';
    chart.setFrequencies(info.sweep.frequencies, info.room_modes);
    chart.setLevels([], [], []);
    chart.setSelected(-1);
    $('cancel-button').disabled = false;
    startPolling();
  } else {
    state.scale = 1e-9;
    updateTimeReadout(info.state);
  }
}

// ----- slice selection -----

function setAxis(axis) {
  state.axis = axis;
  for (const button of document.querySelectorAll('#axis-picker button')) {
    const active = button.dataset.axis === axis;
    button.classList.toggle('active', active);
    button.setAttribute('aria-checked', String(active));
  }
  const slider = $('slice-slider');
  slider.max = state.info.grid.shape[AXES.indexOf(axis)] - 1;
  slider.value = state.indices[axis];
  sliceChanged();
}

for (const button of document.querySelectorAll('#axis-picker button')) {
  button.addEventListener('click', () => setAxis(button.dataset.axis));
}

let sliceTimer = null;
$('slice-slider').addEventListener('input', () => {
  state.indices[state.axis] = Number($('slice-slider').value);
  updateSliceLabel();
  roomView.setSlice(state.axis, state.indices[state.axis]);
  clearTimeout(sliceTimer);
  sliceTimer = setTimeout(sliceChanged, 40);
});

function updateSliceLabel() {
  const index = state.indices[state.axis];
  const metres = (index + 0.5) * state.info.grid.dx;
  $('slice-value').textContent = `${state.axis} = ${metres.toFixed(2)} m (#${index})`;
}

async function sliceChanged() {
  const { axis } = state;
  const index = state.indices[axis];
  updateSliceLabel();
  const token = ++state.geometryToken;
  try {
    const flags = await api(`/api/slice?field=flags&axis=${axis}&index=${index}`);
    if (token !== state.geometryToken) return;
    sliceView.setMeta({
      axis, index, horizontal: flags.horizontal, vertical: flags.vertical,
      dx: state.info.grid.dx, format: formatValue,
    });
    sliceView.setFlags(flags.data, flags.rows, flags.cols);
    roomView.setSlice(axis, index);
    await refreshSliceData();
  } catch (error) {
    showError(error);
  }
}

async function refreshSliceData() {
  const token = ++state.dataToken;
  const { axis } = state;
  const index = state.indices[axis];
  if (state.info.mode === 'sweep') {
    if (state.selected < 0) {
      // nothing simulated yet: show the geometry only
      const empty = await api(`/api/slice?field=flags&axis=${axis}&index=${index}`);
      if (token !== state.dataToken) return;
      sliceView.data = null;
      sliceView.setData(null, empty.rows, empty.cols, sequentialMap(), [0, 1]);
      updateColorbar(null);
      return;
    }
    const slice = await api(`/api/slice?field=leq&axis=${axis}&index=${index}&frequency=${state.selected}`);
    if (token !== state.dataToken) return;
    showLeq(slice);
  } else if (!state.playing) {
    const slice = await api(`/api/slice?field=pressure&axis=${axis}&index=${index}`);
    if (token !== state.dataToken) return;
    showPressure(slice);
  }
}

function formatValue(v) {
  if (Number.isNaN(v)) return 'wall / object';
  if (state.info.mode === 'sweep') return `${v.toFixed(1)} dB`;
  return `${v.toExponential(2)} rel. Pa`;
}

function formatScale(v) {
  if (state.info.mode === 'sweep') return v.toFixed(1);
  if (v === 0) return '0';
  return Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? v.toPrecision(2) : v.toExponential(1);
}

function updateColorbar(range = sliceView.range) {
  $('cb-ramp').style.background = gradientCss(sliceView.map);
  if (!range) {
    $('cb-max').textContent = '';
    $('cb-min').textContent = '';
    return;
  }
  // the Leq range is clipped to percentiles, so its ends are open
  const clipped = state.info.mode === 'sweep';
  $('cb-max').textContent = `${clipped ? '≥ ' : ''}${formatScale(range[1])}`;
  $('cb-min').textContent = `${clipped ? '≤ ' : ''}${formatScale(range[0])}`;
}

// ----- slow motion -----

function showPressure(slice) {
  if (!$('hold-scale').checked) {
    let peak = 0;
    for (const v of slice.data) if (Number.isFinite(v) && Math.abs(v) > peak) peak = Math.abs(v);
    state.scale = peak > 0 ? peak : 1e-9;
  }
  sliceView.setData(slice.data, slice.rows, slice.cols, divergingMap(), [-state.scale, state.scale]);
  updateColorbar();
  if (slice.time !== null) updateTimeReadout({ time: slice.time, iteration: slice.iteration });
}

function updateTimeReadout({ time, iteration }) {
  $('time-readout').textContent = `t = ${(time * 1000).toFixed(3)} ms · step ${iteration}`;
}

function stepSize() {
  const value = Math.round(Number($('step-size').value));
  return Math.min(100000, Math.max(1, Number.isFinite(value) ? value : 1));
}

async function stepOnce() {
  if (state.busy) return;
  state.busy = true;
  const token = ++state.dataToken;
  try {
    const slice = await api('/api/step', {
      method: 'POST',
      body: { count: stepSize(), axis: state.axis, index: state.indices[state.axis] },
    });
    if (token === state.dataToken) showPressure(slice);
  } catch (error) {
    setPlaying(false);
    showError(error);
  } finally {
    state.busy = false;
  }
}

function setPlaying(playing) {
  state.playing = playing;
  $('play-button').textContent = playing ? '❚❚ Pause' : '▶ Play';
  $('step-button').disabled = playing;
  if (playing) playLoop();
}

async function playLoop() {
  while (state.playing && state.info?.mode === 'slowmotion') {
    await stepOnce();
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
}

$('play-button').addEventListener('click', () => setPlaying(!state.playing));
$('step-button').addEventListener('click', () => stepOnce());
$('reset-button').addEventListener('click', async () => {
  setPlaying(false);
  try {
    const status = await api('/api/reset', { method: 'POST' });
    updateTimeReadout(status);
    state.scale = 1e-9;
    await refreshSliceData();
  } catch (error) {
    showError(error);
  }
});
$('hold-scale').addEventListener('change', () => {
  if (!$('hold-scale').checked && !state.playing) refreshSliceData().catch(showError);
});

document.addEventListener('keydown', (event) => {
  if ($('screen-viewer').hidden || state.info?.mode !== 'slowmotion') return;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName) && event.target.type !== 'range') return;
  if (event.code === 'Space') {
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.code === 'ArrowRight' && !state.playing && event.target.type !== 'range') {
    event.preventDefault();
    stepOnce();
  }
});

// ----- sweep -----

function showLeq(slice) {
  sliceView.setData(slice.data, slice.rows, slice.cols, sequentialMap(), [slice.min, slice.max]);
  updateColorbar();
}

function selectFrequency(index) {
  const status = state.sweepStatus;
  if (!status || index < 0 || index >= status.completed) return;
  const changed = index !== state.selected;
  state.selected = index;
  $('freq-slider').value = index;
  const frequency = state.info.sweep.frequencies[index];
  const level = status.spl_avg[index];
  $('freq-value').textContent =
    `${frequency.toFixed(1)} Hz · listeners ${level === null ? '–' : `${level.toFixed(1)} dB`} (${index + 1}/${status.total})`;
  const near = state.info.room_modes
    .filter((m) => Math.abs(m.frequency - frequency) / frequency < 0.04)
    .slice(0, 4);
  $('mode-hint').textContent = near.length
    ? `Analytic box modes nearby: ${near.map(describeMode).join(' · ')}`
    : 'No analytic box mode within 4% of this frequency.';
  chart.setSelected(index);
  if (changed) refreshSliceData().catch(showError);
}

$('freq-slider').addEventListener('input', () => {
  $('follow-latest').checked = false;
  selectFrequency(Number($('freq-slider').value));
});
$('follow-latest').addEventListener('change', () => {
  if ($('follow-latest').checked && state.sweepStatus?.completed) selectFrequency(state.sweepStatus.completed - 1);
});

function startPolling() {
  stopPolling();
  const poll = async () => {
    try {
      const status = await api('/api/sweep');
      applySweepStatus(status);
      if (status.status === 'running') state.pollTimer = setTimeout(poll, 400);
    } catch (error) {
      showError(error);
    }
  };
  poll();
}

function stopPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function applySweepStatus(status) {
  if (state.info?.mode !== 'sweep') return;
  const previous = state.sweepStatus?.completed ?? 0;
  state.sweepStatus = status;
  const percent = Math.round(status.progress * 1000) / 10;
  $('progress-fill').style.width = `${percent}%`;
  $('progress').setAttribute('aria-valuenow', String(percent));
  const frequencies = state.info.sweep.frequencies;
  let text;
  if (status.status === 'running') {
    const current = frequencies[Math.min(status.completed, frequencies.length - 1)];
    text = `${status.completed}/${status.total} · ${current.toFixed(1)} Hz · ${formatDuration(status.remaining)} left`;
  } else if (status.status === 'done') {
    text = `Done: ${status.total} frequencies in ${formatDuration(status.elapsed)}`;
  } else if (status.status === 'cancelled') {
    text = `Cancelled after ${status.completed}/${status.total}`;
  } else {
    text = 'Failed';
    showError(new Error(status.error || 'the sweep failed'));
  }
  $('progress-text').textContent = text;
  $('cancel-button').disabled = status.status !== 'running';

  chart.setLevels(status.spl_avg, status.spl_min, status.spl_max);
  $('freq-slider').max = Math.max(0, status.completed - 1);
  if (status.completed > previous) {
    if ($('follow-latest').checked || state.selected < 0) selectFrequency(status.completed - 1);
  }
}

$('cancel-button').addEventListener('click', async () => {
  try {
    await api('/api/sweep/cancel', { method: 'POST' });
  } catch (error) {
    showError(error);
  }
});

// ---------- start ----------

async function init() {
  try {
    state.options = await api('/api/options');
  } catch (error) {
    showError(error);
    return;
  }
  const { options } = state;
  fillSelect(form.scene, options.scenes.map((s) => ({ value: s.key, label: s.name })));
  fillSelect(form.scheme, options.schemes.map((s) => ({ value: s.key, label: s.label })));
  fillSelect(form.precision, options.precisions.map((p) => ({ value: p, label: p })));
  fillSelect(form.source_type, options.source_types.map((t) => ({ value: t, label: t })));
  fillSelect(form.generator, options.generators.map((g) => ({ value: g, label: g })));
  for (const [key, value] of Object.entries(options.defaults)) {
    if (form[key]) form[key].value = value;
  }
  applySceneDefaults();
  showScreen('mode');

  // resume an existing session after a page reload
  try {
    const info = await api('/api/session');
    // only resume while the user has not started choosing something else
    if (!info.mode || state.mode) return;
    state.mode = info.mode;
    for (const section of document.querySelectorAll('[data-for]')) {
      section.hidden = section.dataset.for !== state.mode;
    }
    await openViewer(info);
  } catch {
    // no session yet
  }
}

init();
