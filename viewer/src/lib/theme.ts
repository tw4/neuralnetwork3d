import { Color } from "three";

/**
 * Renk sozlugu. Referans videodaki paletten cikarildi:
 *   pozitif agirlik -> turkuaz, negatif -> kizil, notr/zayif -> krem beyaz.
 * Aktivasyon parlakligi ayarlar; renk her zaman AGIRLIGIN ISARETINI gosterir.
 * Ikisi ayni kanala binseydi "bu baglanti bastiriyor mu yoksa sessiz mi"
 * ayrimi kaybolurdu.
 */
export const THEME = {
  bg: 0x000000,
  fog: 0x02040a,

  pos: new Color(0x2ee6c0),      // pozitif agirlik
  neg: new Color(0xe8365a),      // negatif agirlik
  idle: new Color(0xd8cfc0),     // agirliksiz/yapisal baglanti (krem)
  recurrent: new Color(0xb46cf0),// RNN kendine donen halka (mor)
  hover: new Color(0xffd75e),    // uzerine gelinen noronun baglantilari

  neuron: new Color(0xe8ecf4),
  neuronHot: new Color(0x9ef7e4),

  card: {
    title: "#e8ecf4",
    dim: "#5d6577",
    act: "#e8365a",
    frame: "rgba(130,165,215,0.30)",
  },

  label: {
    lo: "#4a5265",   // dusuk aktivasyon
    mid: "#e8ecf4",  // orta
    hi: "#2ee6c0",   // yuksek
  },
} as const;

/** Aktivasyon degerini 0..1 parlakliga sikistirir.
 *  ReLU ciktilari sinirsizdir (16.0 gorebiliriz), bu yuzden dogrusal
 *  olcekleme her seyi ya soner ya patlatir; yumusak doyum kullaniyoruz. */
export function intensity(v: number, scale = 1): number {
  const a = Math.abs(v) / Math.max(1e-6, scale);
  return a / (1 + a);
}
