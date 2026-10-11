// routing.js: shortest path from a car to the nearest free slot.
// Coordinates are pixels of the 1080x600 reference frame used by detector.py.
const Router = (() => {
  const W = 1080, H = 600, CELL = 6;
  const GW = Math.ceil(W / CELL), GH = Math.ceil(H / CELL);

  // Asphalt a car can drive on: the aisle in front of the slots, the junction
  // on the right, and the apron at the bottom of the frame.
  const DRIVABLE = [
    [0, 514], [300, 497], [560, 462], [745, 431], [790, 424], [830, 418],
    [872, 396], [915, 390], [1020, 362], [1060, 352], [1080, 360],
    [1080, 415], [815, 600], [0, 600],
  ];

  function inPoly(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function cellOf(x, y) {
    const gx = Math.min(GW - 1, Math.max(0, Math.floor(x / CELL)));
    const gy = Math.min(GH - 1, Math.max(0, Math.floor(y / CELL)));
    return gy * GW + gx;
  }
  const centerOf = (c) => [((c % GW) + 0.5) * CELL, (Math.floor(c / GW) + 0.5) * CELL];

  // Build the walkable grid and per-slot geometry from bounding_boxes.json.
  function build(slots) {
    const walk = new Uint8Array(GW * GH);
    for (let gy = 0; gy < GH; gy++) {
      for (let gx = 0; gx < GW; gx++) {
        const cx = (gx + 0.5) * CELL, cy = (gy + 0.5) * CELL;
        if (!inPoly(cx, cy, DRIVABLE)) continue;
        if (slots.some((s) => inPoly(cx, cy, s.points))) continue;
        walk[gy * GW + gx] = 1;
      }
    }
    const meta = slots.map((s) => {
      const p = s.points;
      const center = [p.reduce((a, q) => a + q[0], 0) / 4, p.reduce((a, q) => a + q[1], 0) / 4];
      // The open end of a slot is the edge between points 3 and 2.
      const mid = [(p[3][0] + p[2][0]) / 2, (p[3][1] + p[2][1]) / 2];
      let dx = mid[0] - center[0], dy = mid[1] - center[1];
      const len = Math.hypot(dx, dy) || 1;
      const access = [mid[0] + (dx / len) * 16, mid[1] + (dy / len) * 16];
      return { center, mid, access };
    });
    const map = { walk, meta };
    meta.forEach((m) => { m.accessCell = nearestWalk(map, m.access[0], m.access[1]); });
    return map;
  }

  // Closest walkable cell to a point (ring search outward).
  function nearestWalk(map, x, y) {
    const c0 = cellOf(x, y);
    if (map.walk[c0]) return c0;
    const gx0 = c0 % GW, gy0 = Math.floor(c0 / GW);
    for (let r = 1; r < Math.max(GW, GH); r++) {
      let best = -1, bd = Infinity;
      for (let gy = gy0 - r; gy <= gy0 + r; gy++) {
        for (let gx = gx0 - r; gx <= gx0 + r; gx++) {
          if (gx < 0 || gy < 0 || gx >= GW || gy >= GH) continue;
          if (Math.max(Math.abs(gx - gx0), Math.abs(gy - gy0)) !== r) continue;
          const c = gy * GW + gx;
          if (!map.walk[c]) continue;
          const d = (gx - gx0) ** 2 + (gy - gy0) ** 2;
          if (d < bd) { bd = d; best = c; }
        }
      }
      if (best >= 0) return best;
    }
    return -1;
  }

  // Minimal binary heap keyed by distance.
  class Heap {
    constructor() { this.k = []; this.v = []; }
    push(key, val) {
      const k = this.k, v = this.v;
      let i = k.length; k.push(key); v.push(val);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (k[p] <= k[i]) break;
        [k[p], k[i]] = [k[i], k[p]]; [v[p], v[i]] = [v[i], v[p]]; i = p;
      }
    }
    pop() {
      const k = this.k, v = this.v;
      const top = v[0], lk = k.pop(), lv = v.pop();
      if (k.length) {
        k[0] = lk; v[0] = lv;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1; let m = i;
          if (l < k.length && k[l] < k[m]) m = l;
          if (r < k.length && k[r] < k[m]) m = r;
          if (m === i) break;
          [k[m], k[i]] = [k[i], k[m]]; [v[m], v[i]] = [v[i], v[m]]; i = m;
        }
      }
      return top;
    }
    get size() { return this.k.length; }
  }

  const NB = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
              [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];

  // Shortest distance (in pixels) from the car to every walkable cell.
  function field(map, x, y) {
    const start = nearestWalk(map, x, y);
    const dist = new Float32Array(GW * GH).fill(Infinity);
    const prev = new Int32Array(GW * GH).fill(-1);
    const heap = new Heap();
    dist[start] = 0; heap.push(0, start);
    const done = new Uint8Array(GW * GH);
    while (heap.size) {
      const c = heap.pop();
      if (done[c]) continue;
      done[c] = 1;
      const gx = c % GW, gy = (c / GW) | 0;
      for (const [dx, dy, w] of NB) {
        const nx = gx + dx, ny = gy + dy;
        if (nx < 0 || ny < 0 || nx >= GW || ny >= GH) continue;
        const n = ny * GW + nx;
        if (!map.walk[n]) continue;
        if (dx && dy && (!map.walk[gy * GW + nx] || !map.walk[ny * GW + gx])) continue; // no corner cutting
        const nd = dist[c] + w * CELL;
        if (nd < dist[n]) { dist[n] = nd; prev[n] = c; heap.push(nd, n); }
      }
    }
    return { start, dist, prev };
  }

  // Cost to reach a slot centre: drive to its access point, then pull in.
  function costTo(map, f, i) {
    const m = map.meta[i];
    if (m.accessCell < 0 || !isFinite(f.dist[m.accessCell])) return Infinity;
    const [cx, cy] = centerOf(m.accessCell);
    return f.dist[m.accessCell] + Math.hypot(m.access[0] - cx, m.access[1] - cy) +
      Math.hypot(m.center[0] - m.access[0], m.center[1] - m.access[1]);
  }

  // Pick the nearest free slot. Keeps the current target unless another slot is
  // clearly closer, so the guidance does not flicker between near-equal slots.
  function nearest(map, f, freeList, current) {
    let best = -1, bc = Infinity;
    for (const i of freeList) {
      const c = costTo(map, f, i);
      if (c < bc) { bc = c; best = i; }
    }
    if (best < 0) return null;
    if (current != null && current !== best && freeList.includes(current)) {
      const cc = costTo(map, f, current);
      if (cc <= bc + 24) return { slot: current, cost: cc };
    }
    return { slot: best, cost: bc };
  }

  function clear(map, a, b) {
    const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 3);
    for (let i = 0; i <= n; i++) {
      const t = n ? i / n : 0;
      const c = cellOf(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
      if (!map.walk[c]) return false;
    }
    return true;
  }

  // Polyline from the car to the slot centre, with grid zig-zags straightened.
  function route(map, f, i, from) {
    const m = map.meta[i];
    const cells = [];
    for (let c = m.accessCell; c !== -1; c = f.prev[c]) cells.push(c);
    cells.reverse();
    const pts = cells.map(centerOf);
    pts[0] = [from[0], from[1]];
    const out = [pts[0]];
    let a = 0;
    while (a < pts.length - 1) {
      let b = pts.length - 1;
      while (b > a + 1 && !clear(map, pts[a], pts[b])) b--;
      out.push(pts[b]);
      a = b;
    }
    out.push(m.access, m.center);
    return out;
  }

  function length(pts) {
    let s = 0;
    for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    return s;
  }

  return { W, H, CELL, DRIVABLE, inPoly, build, field, costTo, nearest, route, length, nearestWalk, centerOf };
})();
if (typeof module !== 'undefined') module.exports = Router;
