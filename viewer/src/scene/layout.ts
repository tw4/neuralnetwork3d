import { Vector3 } from "three";
import type { Graph, Layer } from "../lib/types";

export interface PlacedLayer {
  layer: Layer;
  /** Ekranda gercekten cizilen nokta sayisi (size degil!). */
  count: number;
  /** Her noktanin dunya konumu. */
  points: Vector3[];
  x: number;
  height: number;
  /** Global nokta indeksinin basladigi yer (tek bir Points tamponu kullaniyoruz). */
  offset: number;
}

export interface Layout {
  layers: PlacedLayer[];
  byId: Map<string, PlacedLayer>;
  totalPoints: number;
  width: number;
  maxHeight: number;
}

// Bu uc sayi goruntunun karakterini belirliyor. Kavis genligi artik yatay
// aralik yerine sutun yuksekligine bagli oldugu icin (bkz. edges.ts) araligi
// rahatca acabiliyoruz: katmanlar ust uste binmiyor ama demetler yine yogun.
// GAP'i genis tutmanin ikinci bir sebebi daha var: kamera agi hem dikey hem
// yatay olarak sigdiriyor, yani hangi eksen daha "dolu" ise o baglayici
// oluyor. Dar arali bir agin en-boy orani genis ekrandan kucuk kalinca dikey
// eksen baglayici oluyor ve saglar/sollar bos kaliyordu.
const GAP = 17;           // katmanlar arasi yatay mesafe
const SPACING = 1.5;      // ayni katmandaki noronlar arasi dikey mesafe
const PASSTHRU_GAP = 10;  // Dropout/BatchNorm gibi katmanlar daha yakin dursun

/**
 * Katmanlari soldan saga dizer, noronlari dikey sutunlara yerlestirir.
 *
 * Neden hepsi tek duzlemde (z=0) degil: tamamen duz bir dizilim, kamera
 * dondugunde derinlik hissi vermiyor ve kenarlar ust uste binip lapa
 * gorunuyordu. Sutunlara cok hafif bir z salinimi veriyoruz -- gozle zor
 * secilir ama 3B'de katmanlarin ayrismasini saglar.
 */
export function computeLayout(graph: Graph): Layout {
  const placed: PlacedLayer[] = [];
  let x = 0;
  let offset = 0;
  let maxHeight = 0;

  graph.layers.forEach((layer, i) => {
    const count = Math.max(1, Math.min(layer.size, layer.display.maxNeurons));
    const height = (count - 1) * SPACING;
    maxHeight = Math.max(maxHeight, height);

    const zWave = Math.sin(i * 0.9) * 0.6;
    const points: Vector3[] = [];
    for (let n = 0; n < count; n++) {
      const y = height / 2 - n * SPACING;
      points.push(new Vector3(x, y, zWave));
    }

    placed.push({ layer, count, points, x, height, offset });
    offset += count;

    const next = graph.layers[i + 1];
    x += next && (next.kind === "passthrough" || layer.kind === "passthrough")
      ? PASSTHRU_GAP
      : GAP;
  });

  // Butun agi orijinde ortala ki kamera hep ayni yere baksin.
  const width = placed.length ? placed[placed.length - 1].x : 0;
  for (const p of placed) {
    p.x -= width / 2;
    for (const v of p.points) v.x -= width / 2;
  }

  return {
    layers: placed,
    byId: new Map(placed.map((p) => [p.layer.id, p])),
    totalPoints: offset,
    width,
    maxHeight,
  };
}
