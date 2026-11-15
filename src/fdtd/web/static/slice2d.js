// 2D slice rendering: colour maps, walls, source / listener overlays and hover.

const FLAG_WALL = 1;
const FLAG_SOURCE = 2;
const FLAG_LISTENER = 4;

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function isDark() {
  const theme = document.documentElement.dataset.theme;
  if (theme) return theme === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

// Piecewise linear colour map over stops [[t, '#hex'], ...], t in [0, 1]
function makeMap(stops) {
  const parsed = stops.map(([t, hex]) => [t, hexToRgb(hex)]);
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < parsed.length - 2 && t > parsed[k + 1][0]) k++;
    const [t0, c0] = parsed[k];
    const [t1, c1] = parsed[k + 1];
    const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
    for (let ch = 0; ch < 3; ch++) lut[i * 3 + ch] = c0[ch] + (c1[ch] - c0[ch]) * f;
  }
  return { stops, lut };
}

// Diverging blue <-> red with the neutral surface grey at zero pressure
export function divergingMap() {
  const neutral = isDark() ? '#383835' : '#f0efec';
  return makeMap([[0, '#0d366b'], [0.25, '#3987e5'], [0.5, neutral], [0.75, '#e66767'], [1, '#8f1d1d']]);
}

// Sequential single hue (blue ramp): low level recedes towards the surface
export function sequentialMap() {
  const ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
  if (isDark()) ramp.reverse();
  return makeMap(ramp.map((hex, i) => [i / (ramp.length - 1), hex]));
}

export function gradientCss(map) {
  // vertical colour bar: max on top
  const parts = map.stops.map(([t, hex]) => `${hex} ${(100 - t * 100).toFixed(1)}%`).reverse();
  return `linear-gradient(to bottom, ${parts.join(', ')})`;
}

export class SliceView {
  constructor({ box, canvas, tooltip, axisX, axisY, onResize }) {
    this.box = box;
    this.canvas = canvas;
    this.tooltip = tooltip;
    this.axisX = axisX;
    this.axisY = axisY;
    // cell image: one pixel per grid cell, also used as the 3D plane texture
    this.image = document.createElement('canvas');
    this.imageCtx = this.image.getContext('2d');
    this.rows = 0;
    this.cols = 0;
    this.data = null;
    this.flags = null;
    this.map = divergingMap();
    this.range = [-1, 1];
    this.meta = null;  // {axis, index, horizontal, vertical, dx, unit, format}
    this.onImageChange = null;

    new ResizeObserver(() => { this.layout(); onResize?.(); }).observe(box);
    canvas.addEventListener('mousemove', (e) => this.hover(e));
    canvas.addEventListener('mouseleave', () => { this.tooltip.hidden = true; });
  }

  setMeta(meta) {
    this.meta = meta;
    const names = { w: 'width', h: 'height', d: 'depth' };
    this.axisX.textContent = `${names[meta.horizontal]} (${meta.horizontal}) →`;
    this.axisY.textContent = meta.vertical === 'h' ? `${names.h} (h) →` : `← ${names.d} (d)`;
  }

  setFlags(flags, rows, cols) {
    this.flags = flags;
    this.flagRows = rows;
    this.flagCols = cols;
  }

  setData(data, rows, cols, map, range) {
    this.data = data;
    this.map = map;
    this.range = range;
    if (rows !== this.rows || cols !== this.cols) {
      this.rows = rows;
      this.cols = cols;
      this.image.width = cols;
      this.image.height = rows;
      this.layout();
    }
    this.paintImage();
    this.draw();
  }

  clear() {
    this.data = null;
    this.paintImage();
    this.draw();
  }

  paintImage() {
    const { rows, cols, data, flags } = this;
    if (!rows || !cols) return;
    const pixels = this.imageCtx.createImageData(cols, rows);
    const out = pixels.data;
    const [low, high] = this.range;
    const span = high - low || 1;
    const lut = this.map.lut;
    const voidRgb = hexToRgb(cssVar('--void') || '#3a3936');
    const emptyRgb = hexToRgb(cssVar('--surface-2') || '#f0efec');
    const sameShape = flags && this.flagRows === rows && this.flagCols === cols;
    for (let i = 0; i < rows * cols; i++) {
      const v = data ? data[i] : 0;
      const wall = sameShape ? (flags[i] & FLAG_WALL) : Number.isNaN(v);
      let r, g, b;
      if (wall || Number.isNaN(v)) {
        [r, g, b] = voidRgb;
      } else if (!data) {
        [r, g, b] = emptyRgb;
      } else {
        const t = Math.min(255, Math.max(0, Math.round(((v - low) / span) * 255)));
        r = lut[t * 3]; g = lut[t * 3 + 1]; b = lut[t * 3 + 2];
      }
      out[i * 4] = r; out[i * 4 + 1] = g; out[i * 4 + 2] = b; out[i * 4 + 3] = 255;
    }
    this.imageCtx.putImageData(pixels, 0, 0);
    this.onImageChange?.();
  }

  layout() {
    const { rows, cols } = this;
    if (!rows || !cols) return;
    const pad = 18;  // room for the axis labels
    const width = this.box.clientWidth - pad;
    const height = this.box.clientHeight - pad;
    const scale = Math.max(0.1, Math.min(width / cols, height / rows));
    const cssW = cols * scale;
    const cssH = rows * scale;
    const ratio = window.devicePixelRatio || 1;
    this.scale = scale;
    this.canvas.style.width = `${cssW}px`;
    this.canvas.style.height = `${cssH}px`;
    this.canvas.style.left = `${pad + (width - cssW) / 2}px`;
    this.canvas.style.top = `${(height - cssH) / 2}px`;
    this.canvas.width = Math.max(1, Math.round(cssW * ratio));
    this.canvas.height = Math.max(1, Math.round(cssH * ratio));
    this.draw();
  }

  draw() {
    const ctx = this.canvas.getContext('2d');
    const { rows, cols, flags } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!rows || !cols) return;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
    if (!flags || this.flagRows !== rows || this.flagCols !== cols) return;

    const sx = this.canvas.width / cols;
    const sy = this.canvas.height / rows;
    const ratio = window.devicePixelRatio || 1;

    // listener regions: outline of the region boundary
    ctx.strokeStyle = cssVar('--listener');
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = Math.max(1, ratio);
    ctx.beginPath();
    const isListener = (r, c) => r >= 0 && c >= 0 && r < rows && c < cols && (flags[r * cols + c] & FLAG_LISTENER);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (!isListener(r, c)) continue;
        const x = c * sx, y = r * sy;
        if (!isListener(r - 1, c)) { ctx.moveTo(x, y); ctx.lineTo(x + sx, y); }
        if (!isListener(r + 1, c)) { ctx.moveTo(x, y + sy); ctx.lineTo(x + sx, y + sy); }
        if (!isListener(r, c - 1)) { ctx.moveTo(x, y); ctx.lineTo(x, y + sy); }
        if (!isListener(r, c + 1)) { ctx.moveTo(x + sx, y); ctx.lineTo(x + sx, y + sy); }
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // the active source cell: ring with a surface halo
    for (let i = 0; i < rows * cols; i++) {
      if (!(flags[i] & FLAG_SOURCE)) continue;
      const cx = (i % cols + 0.5) * sx;
      const cy = (Math.floor(i / cols) + 0.5) * sy;
      const radius = Math.max(5 * ratio, Math.min(sx, sy) * 0.8);
      ctx.lineWidth = 4 * ratio;
      ctx.strokeStyle = cssVar('--surface');
      ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
      ctx.lineWidth = 2 * ratio;
      ctx.strokeStyle = cssVar('--source');
      ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // grid cell (w, h, d) under a display (row, col)
  cellAt(row, col) {
    const m = this.meta;
    const cell = { w: 0, h: 0, d: 0 };
    cell[m.axis] = m.index;
    cell[m.horizontal] = col;
    cell[m.vertical] = m.vertical === 'h' ? this.rows - 1 - row : row;
    return cell;
  }

  hover(event) {
    if (!this.meta || !this.rows) return;
    const rect = this.canvas.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * this.cols);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * this.rows);
    if (row < 0 || col < 0 || row >= this.rows || col >= this.cols) {
      this.tooltip.hidden = true;
      return;
    }
    const cell = this.cellAt(row, col);
    const dx = this.meta.dx;
    const pos = ['w', 'h', 'd'].map((a) => `${a} ${((cell[a] + 0.5) * dx).toFixed(2)}`).join('  ');
    const i = row * this.cols + col;
    const flags = this.flags && this.flagRows === this.rows ? this.flags[i] : 0;
    let value;
    if (flags & FLAG_WALL) value = 'wall / object';
    else if (!this.data) value = '–';
    else value = this.meta.format(this.data[i]);
    const extra = [];
    if (flags & FLAG_SOURCE) extra.push('source');
    if (flags & FLAG_LISTENER) extra.push('listener');
    this.tooltip.innerHTML = `<b>${value}</b>${extra.length ? ` · ${extra.join(', ')}` : ''}<br>${pos} m`;
    this.tooltip.hidden = false;
    const boxRect = this.box.getBoundingClientRect();
    let x = event.clientX - boxRect.left + 14;
    const y = event.clientY - boxRect.top + 14;
    if (x + this.tooltip.offsetWidth > this.box.clientWidth) x -= this.tooltip.offsetWidth + 28;
    this.tooltip.style.left = `${x}px`;
    this.tooltip.style.top = `${y}px`;
  }
}
