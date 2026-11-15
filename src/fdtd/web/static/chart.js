// Listener level per frequency (log x), with min-max band and analytic room modes.

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const TICKS = [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 800,
  1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000];
const MODE_DASH = { axial: [], tangential: [5, 4], oblique: [1.5, 3] };

function formatHz(f) {
  return f >= 1000 ? `${(f / 1000).toFixed(f % 1000 ? 1 : 0)}k` : `${Math.round(f)}`;
}

export function describeMode(mode) {
  return `(${mode.order.join(',')}) ${mode.frequency.toFixed(1)} Hz ${mode.kind}`;
}

export class SweepChart {
  constructor(canvas, tooltip, onSelect) {
    this.canvas = canvas;
    this.tooltip = tooltip;
    this.onSelect = onSelect;
    this.frequencies = [];
    this.avg = [];
    this.min = [];
    this.max = [];
    this.modes = [];
    this.selected = -1;
    this.hoverIndex = -1;
    this.margin = { left: 44, right: 12, top: 10, bottom: 24 };

    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener('mousemove', (e) => this.hover(e));
    canvas.addEventListener('mouseleave', () => {
      this.hoverIndex = -1;
      this.tooltip.hidden = true;
      this.draw();
    });
    canvas.addEventListener('click', (e) => {
      const index = this.indexAt(e);
      if (index >= 0) this.onSelect?.(index);
    });
  }

  setFrequencies(frequencies, modes) {
    this.frequencies = frequencies;
    this.modes = modes;
    this.draw();
  }

  setLevels(avg, min, max) {
    this.avg = avg;
    this.min = min;
    this.max = max;
    this.draw();
  }

  setSelected(index) {
    this.selected = index;
    this.draw();
  }

  geometry() {
    const rect = this.canvas.getBoundingClientRect();
    const { left, right, top, bottom } = this.margin;
    const f = this.frequencies;
    const fLow = f[0] / 1.06;
    const fHigh = f[f.length - 1] * 1.06;
    const values = [...this.min, ...this.max].filter((v) => v !== null && Number.isFinite(v));
    let yLow = values.length ? Math.min(...values) : 0;
    let yHigh = values.length ? Math.max(...values) : 100;
    const step = (yHigh - yLow) > 40 ? 10 : 5;
    yLow = Math.floor(yLow / step) * step;
    yHigh = Math.max(yLow + step, Math.ceil(yHigh / step) * step);
    const plotW = rect.width - left - right;
    const plotH = rect.height - top - bottom;
    return {
      rect, step, yLow, yHigh, fLow, fHigh, plotW, plotH,
      x: (freq) => left + (Math.log(freq / fLow) / Math.log(fHigh / fLow)) * plotW,
      y: (db) => top + (1 - (db - yLow) / (yHigh - yLow)) * plotH,
    };
  }

  draw() {
    const canvas = this.canvas;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!this.frequencies.length || !rect.width) return;

    const g = this.geometry();
    const { left, top } = this.margin;
    const ink2 = cssVar('--ink-2');
    const muted = cssVar('--muted');
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';

    // horizontal gridlines + y labels
    ctx.strokeStyle = cssVar('--grid');
    ctx.fillStyle = muted;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let v = g.yLow; v <= g.yHigh + 1e-9; v += g.step) {
      const y = Math.round(g.y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + g.plotW, y); ctx.stroke();
      ctx.fillText(`${v}`, left - 6, y);
    }
    ctx.save();
    ctx.translate(10, top + g.plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('dB (Leq)', 0, 0);
    ctx.restore();

    // x ticks
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const ticks = TICKS.filter((t) => t >= g.fLow && t <= g.fHigh);
    const minGap = 30;
    let lastX = -Infinity;
    for (const t of ticks) {
      const x = g.x(t);
      if (x - lastX < minGap) continue;
      lastX = x;
      ctx.fillText(formatHz(t), x, top + g.plotH + 6);
    }
    ctx.textAlign = 'right';
    ctx.fillText('Hz', left + g.plotW, top + g.plotH + 6 + 0);

    // baseline
    ctx.strokeStyle = cssVar('--axis');
    ctx.beginPath();
    ctx.moveTo(left, top + g.plotH + 0.5);
    ctx.lineTo(left + g.plotW, top + g.plotH + 0.5);
    ctx.stroke();

    // analytic room modes
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, g.plotW, g.plotH);
    ctx.clip();
    ctx.strokeStyle = muted;
    ctx.globalAlpha = this.modes.length > 120 ? 0.25 : 0.55;
    ctx.lineWidth = 1;
    for (const mode of this.modes) {
      if (mode.frequency < g.fLow || mode.frequency > g.fHigh) continue;
      const x = Math.round(g.x(mode.frequency)) + 0.5;
      ctx.setLineDash(MODE_DASH[mode.kind]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + g.plotH); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    const n = this.avg.length;
    const f = this.frequencies;
    if (n > 0) {
      // min-max band
      ctx.fillStyle = cssVar('--band-1');
      ctx.beginPath();
      for (let i = 0; i < n; i++) ctx.lineTo(g.x(f[i]), g.y(this.max[i] ?? 0));
      for (let i = n - 1; i >= 0; i--) ctx.lineTo(g.x(f[i]), g.y(this.min[i] ?? 0));
      ctx.closePath();
      ctx.fill();

      // average
      ctx.strokeStyle = cssVar('--series-1');
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        if (this.avg[i] === null) continue;
        ctx.lineTo(g.x(f[i]), g.y(this.avg[i]));
      }
      ctx.stroke();
      if (n === 1) this.dot(ctx, g.x(f[0]), g.y(this.avg[0]), cssVar('--series-1'));
    }
    ctx.restore();

    // selected frequency and hover crosshair
    const marker = (index, color, width) => {
      if (index < 0 || index >= f.length) return;
      const x = Math.round(g.x(f[index])) + 0.5;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + g.plotH); ctx.stroke();
      if (index < n && this.avg[index] !== null) this.dot(ctx, g.x(f[index]), g.y(this.avg[index]), cssVar('--series-1'));
    };
    if (this.hoverIndex >= 0 && this.hoverIndex !== this.selected) marker(this.hoverIndex, cssVar('--axis'), 1);
    marker(this.selected, ink2, 1.5);
  }

  dot(ctx, x, y, color) {
    ctx.fillStyle = cssVar('--surface');
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
  }

  indexAt(event) {
    if (!this.frequencies.length) return -1;
    const g = this.geometry();
    const x = event.clientX - g.rect.left;
    let best = -1;
    let bestDistance = Infinity;
    this.frequencies.forEach((freq, i) => {
      const distance = Math.abs(g.x(freq) - x);
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    });
    return bestDistance < 24 ? best : -1;
  }

  hover(event) {
    const index = this.indexAt(event);
    this.hoverIndex = index;
    this.draw();
    if (index < 0) {
      this.tooltip.hidden = true;
      return;
    }
    const freq = this.frequencies[index];
    const fmt = (v) => (v === null || v === undefined ? '–' : `${v.toFixed(1)} dB`);
    const nearest = this.modes.length
      ? this.modes.reduce((a, b) => (Math.abs(b.frequency - freq) < Math.abs(a.frequency - freq) ? b : a))
      : null;
    const lines = [`<b>${freq.toFixed(1)} Hz</b>`];
    if (index < this.avg.length) {
      lines.push(`average ${fmt(this.avg[index])}`);
      lines.push(`min ${fmt(this.min[index])} · max ${fmt(this.max[index])}`);
    } else {
      lines.push('not simulated yet');
    }
    if (nearest) lines.push(`nearest mode ${describeMode(nearest)}`);
    this.tooltip.innerHTML = lines.join('<br>');
    this.tooltip.hidden = false;
    const box = this.canvas.parentElement.getBoundingClientRect();
    let x = event.clientX - box.left + 14;
    if (x + this.tooltip.offsetWidth > box.width) x -= this.tooltip.offsetWidth + 28;
    this.tooltip.style.left = `${x}px`;
    this.tooltip.style.top = `${event.clientY - box.top - 10}px`;
  }
}
