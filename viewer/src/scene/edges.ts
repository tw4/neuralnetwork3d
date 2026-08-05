import {
  AdditiveBlending, BufferAttribute, BufferGeometry, LineBasicMaterial,
  LineSegments, Vector3,
} from "three";
import { THEME, intensity } from "../lib/theme";
import type { Layout, PlacedLayer } from "./layout";
import type { Edge, Graph } from "../lib/types";
import { unpackF32 } from "../lib/types";

/** Bir kenar kac duz parcaya bolunerek cizilir. 14 gozle puruzsuz gorunur. */
const SEG = 14;

interface EdgeRef {
  src: PlacedLayer; dst: PlacedLayer;
  si: number; di: number;      // katman ici noron indisleri
  w: number;                   // agirlik (-inf..inf), 0 ise cizilmez
  vtx: number;                 // bu kenarin renk tamponundaki ilk kose indisi
  kind: Edge["kind"];
}

/**
 * Butun kenarlari TEK bir LineSegments icinde tutar.
 *
 * Neden tek geometri: her kenar icin ayri bir Line nesnesi olusturmak
 * binlerce draw call demek olurdu ve 3072 genisligindeki bir katmanda
 * tarayici kilitlenirdi. Tek tampon + vertex renkleri ile her karede sadece
 * renk dizisini guncelliyoruz; geometri hic degismiyor.
 */
export class EdgeField {
  readonly object: LineSegments;
  private refs: EdgeRef[] = [];
  private colors: Float32Array;
  private geom: BufferGeometry;
  /** noron anahtari ("layerId:index") -> o norona degen kenarlar */
  private touching = new Map<string, EdgeRef[]>();
  private maxAbsW = 1;

  constructor(graph: Graph, layout: Layout) {
    const positions: number[] = [];
    let vtx = 0;

    for (const e of graph.edges) {
      const src = layout.byId.get(e.from);
      const dst = layout.byId.get(e.to);
      if (!src || !dst) continue;

      if (e.kind === "recurrent") {
        vtx = this.buildRecurrent(src, positions, vtx);
        continue;
      }

      const pairs = this.pairsFor(e, src, dst);
      for (const [si, di, w] of pairs) {
        if (w === 0 && e.kind === "dense") continue;   // budanmis agirlik
        const a = src.points[si];
        const b = dst.points[di];
        if (!a || !b) continue;
        // identity baglanti noron i'yi noron i'ye gecirir; onu da savurmak
        // ortada anlamsiz bir "goz" deseni uretiyordu. Neredeyse duz cizsin.
        const bowScale = e.kind === "identity" ? 0.06 : 1;
        appendCurve(positions, a, b, si, di, Math.max(src.height, dst.height), bowScale);
        this.refs.push({ src, dst, si, di, w, vtx, kind: e.kind });
        vtx += SEG * 2;
      }
    }

    for (const r of this.refs) {
      push(this.touching, `${r.src.layer.id}:${r.si}`, r);
      push(this.touching, `${r.dst.layer.id}:${r.di}`, r);
    }

    // Parlaklik olcegi: en buyuk agirlik DEGIL, 75. yuzdelik.
    // Tek bir aykiri agirlik (egitim sirasinda sik olur) maksimumu yukari
    // cekip butun sahneyi karartiyordu; yuzdelik buna bagisik.
    const mags = this.refs
      .filter((r) => !Number.isNaN(r.w))
      .map((r) => Math.abs(r.w))
      .sort((a, b) => a - b);
    this.maxAbsW = mags.length ? Math.max(1e-6, mags[Math.floor(mags.length * 0.75)]) : 1;

    this.colors = new Float32Array(vtx * 3);
    this.geom = new BufferGeometry();
    this.geom.setAttribute("position", new BufferAttribute(new Float32Array(positions), 3));
    this.geom.setAttribute("color", new BufferAttribute(this.colors, 3));

    this.object = new LineSegments(
      this.geom,
      new LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        blending: AdditiveBlending,   // ust uste binen kenarlar parlasin
        depthWrite: false,            // saydamlik sirasi bozulmasin
      })
    );
    this.object.frustumCulled = false;
    this.paint(null, null);
  }

  /** Bir kenar tanimindan (si, di, agirlik) uclulerini cikarir. */
  private pairsFor(e: Edge, src: PlacedLayer, dst: PlacedLayer): [number, number, number][] {
    const out: [number, number, number][] = [];
    if (e.kind === "dense" && e.weights && e.shape) {
      const w = unpackF32(e.weights);
      const [rows, cols] = e.shape;
      for (let i = 0; i < rows && i < src.count; i++) {
        for (let j = 0; j < cols && j < dst.count; j++) {
          out.push([i, j, w[i * cols + j]]);
        }
      }
    } else {
      // identity / reshape: yapisal baglanti, agirlik yok.
      const n = Math.max(src.count, dst.count);
      for (let k = 0; k < n; k++) {
        out.push([k % src.count, k % dst.count, NaN]);
      }
    }
    return out;
  }

  /** RNN'in kendine donen baglantisi: sutunun yaninda kucuk halkalar. */
  private buildRecurrent(l: PlacedLayer, positions: number[], vtx: number): number {
    const R = 0.42;
    for (let i = 0; i < l.count; i++) {
      const p = l.points[i];
      for (let s = 0; s < SEG; s++) {
        const t0 = (s / SEG) * Math.PI * 2;
        const t1 = ((s + 1) / SEG) * Math.PI * 2;
        positions.push(p.x - R + Math.cos(t0) * R, p.y + Math.sin(t0) * R, p.z);
        positions.push(p.x - R + Math.cos(t1) * R, p.y + Math.sin(t1) * R, p.z);
      }
      this.refs.push({ src: l, dst: l, si: i, di: i, w: NaN, vtx, kind: "recurrent" });
      vtx += SEG * 2;
    }
    return vtx;
  }

  /**
   * Renkleri yeniden hesaplar.
   * @param acts katman id -> aktivasyon vektoru (yoksa sadece agirliklar gorunur)
   * @param hover uzerine gelinen noron anahtari ("layerId:index")
   */
  paint(acts: Map<string, Float32Array> | null, hover: string | null): void {
    const hoverSet = hover ? new Set(this.touching.get(hover) ?? []) : null;
    const c = this.colors;

    for (const r of this.refs) {
      let col = THEME.idle;
      let amp: number;

      if (r.kind === "recurrent") {
        col = THEME.recurrent;
        amp = 0.5;
      } else if (Number.isNaN(r.w)) {
        amp = 0.45;                                  // yapisal baglanti: soluk
      } else {
        col = r.w >= 0 ? THEME.pos : THEME.neg;
        amp = 0.35 + 0.95 * intensity(r.w, this.maxAbsW);
      }

      // Kaynak noron ates ediyorsa baglanti parlar -- "sinyal buradan akti".
      if (acts) {
        const a = acts.get(r.src.layer.id);
        if (a && r.si < a.length) amp *= 0.55 + 1.1 * intensity(a[r.si]);
      }

      if (hoverSet) {
        if (hoverSet.has(r)) { col = THEME.hover; amp = 1; }
        else amp *= 0.07;
      }

      const rr = col.r * amp, gg = col.g * amp, bb = col.b * amp;
      const end = (r.vtx + SEG * 2) * 3;
      for (let o = r.vtx * 3; o < end; o += 3) {
        c[o] = rr; c[o + 1] = gg; c[o + 2] = bb;
      }
    }
    (this.geom.getAttribute("color") as BufferAttribute).needsUpdate = true;
  }

  get count(): number { return this.refs.length; }
}

/**
 * Iki noron arasina kubik Bezier kavis cizer.
 *
 * Duz cizgi kullanmiyoruz: tam bagli bir katmanda butun cizgiler ayni dar
 * koridordan gecip birbirini yiyor, sonuc gri bir lekeye donuyor. Kontrol
 * noktalarini disari savurunca cizgiler ayrisiyor ve referanstaki o badem
 * seklindeki hacimli goruntu ortaya cikiyor.
 */
function appendCurve(
  out: number[], a: Vector3, b: Vector3,
  si: number, di: number, colHeight: number, bowScale = 1
): void {
  const dx = b.x - a.x;
  const h = hash2(si * 131 + 17, di * 197 + 31);
  const amp = Math.max(Math.abs(dx) * 0.22, colHeight * 0.30) * bowScale;
  const bow = (h * 2 - 1) * amp;
  const zbow = (hash2(di * 71, si * 53) * 2 - 1) * amp * 0.7;

  const c1x = a.x + dx * 0.30, c1y = a.y + bow, c1z = a.z + zbow;
  const c2x = a.x + dx * 0.70, c2y = b.y + bow, c2z = b.z + zbow;

  let px = a.x, py = a.y, pz = a.z;
  for (let s = 1; s <= SEG; s++) {
    const t = s / SEG;
    const u = 1 - t;
    const w0 = u * u * u, w1 = 3 * u * u * t, w2 = 3 * u * t * t, w3 = t * t * t;
    const x = w0 * a.x + w1 * c1x + w2 * c2x + w3 * b.x;
    const y = w0 * a.y + w1 * c1y + w2 * c2y + w3 * b.y;
    const z = w0 * a.z + w1 * c1z + w2 * c2z + w3 * b.z;
    out.push(px, py, pz, x, y, z);
    px = x; py = y; pz = z;
  }
}

function hash2(i: number, j: number): number {
  const s = Math.sin(i * 12.9898 + j * 78.233) * 43758.5453;
  return s - Math.floor(s);
}

function push<K, V>(m: Map<K, V[]>, k: K, v: V): void {
  const arr = m.get(k);
  if (arr) arr.push(v); else m.set(k, [v]);
}
