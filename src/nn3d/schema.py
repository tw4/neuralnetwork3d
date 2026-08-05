"""nn3d veri sozlesmesi.

Python tarafi ile tarayicidaki 3D cizici arasindaki TEK baglanti burasi.
Cizici modelin ne oldugunu bilmez; sadece bu semayi bilir. Bu sayede ayni
kod hem 10 noronluk bir oyun ajanini hem 12 blokluk bir transformer'i cizer.

Iki mesaj tipi var:

1) GRAFIK -- bir kez gonderilir, agin yapisi
   {
     "version": 1,
     "name": "churn_model",
     "layers": [Layer, ...],
     "edges":  [Edge, ...],
     "groups": [Group, ...]
   }

2) KARE (frame) -- her adimda gonderilir, o anki aktivasyonlar
   {
     "step": 42, "epoch": 1,
     "metrics": {"loss": 0.31, "accuracy": 0.88},
     "act": {"dense_1": "<base64 float32>", ...}
   }

Aktivasyonlar base64'lenmis little-endian float32 dizileridir. JSON sayi
listesi degil; 3072 genislikte bir katmanda aradaki fark 10 kattan fazla.
"""

from __future__ import annotations

import base64
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = 1

# Katman turu -> cizicinin nasil yerlestirecegi.
#   input       : solda, disaridan metin etiketli (Hunger Level, Food Distance...)
#   dense       : klasik tam bagli sutun
#   conv        : kanal basina bir nokta, kart uzerinde uzamsal boyut yazar
#   rnn         : dense gibi ama kendine donen kavis cizilir
#   reshape     : Flatten/Reshape -- veri yeniden duzenlenir, agirlik yok
#   passthrough : Dropout/BatchNorm/Activation -- ince, bagimsiz sutun
#   output      : sagda, sinif etiketli
KINDS = ("input", "dense", "conv", "rnn", "reshape", "passthrough", "output")

# Bir katmanda en fazla kac nokta cizilir. 3072 noronu tek tek cizmek
# hem okunmaz hem 2.3 milyon kenar demek; temsilci noktalara indiriyoruz.
DEFAULT_MAX_NEURONS = 16


def layer(
    id: str,
    label: str,
    kind: str,
    size: int,
    *,
    activation: Optional[str] = None,
    shape_text: Optional[str] = None,
    neuron_labels: Optional[Sequence[str]] = None,
    max_neurons: int = DEFAULT_MAX_NEURONS,
    params: int = 0,
) -> Dict[str, Any]:
    """Tek bir katmani tarif eder.

    size      : gercek noron/kanal sayisi (kart uzerinde bu yazar)
    max_neurons: kac tanesinin cizilecegi (sanallastirma)
    """
    if kind not in KINDS:
        raise ValueError(f"bilinmeyen katman turu: {kind!r} (gecerli: {KINDS})")
    out: Dict[str, Any] = {
        "id": id,
        "label": label,
        "kind": kind,
        "size": int(size),
        "params": int(params),
        "display": {"maxNeurons": int(max_neurons)},
    }
    if activation:
        out["activation"] = activation
    if shape_text:
        out["shapeText"] = shape_text
    if neuron_labels:
        out["neuronLabels"] = list(neuron_labels)
    return out


def edge(
    src: str,
    dst: str,
    *,
    kind: str = "dense",
    shape: Optional[Sequence[int]] = None,
    weights: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """Iki katman arasindaki baglantiyi tarif eder.

    kind="identity" ise agirlik yoktur (Dropout, BatchNorm gibi) ve cizici
    noron i'yi noron i'ye duz cizgiyle baglar.

    weights: satir-oncelikli (row-major) shape[0]*shape[1] uzunlugunda,
    ZATEN sanallastirilmis matris. Isareti renk, buyuklugu parlaklik olur.
    """
    out: Dict[str, Any] = {"from": src, "to": dst, "kind": kind}
    if shape is not None:
        out["shape"] = [int(s) for s in shape]
    if weights is not None:
        vals = [float(w) for w in weights]
        out["weights"] = pack_f32(vals)
        out["weightsCount"] = len(vals)
    return out


def group(label: str, layer_ids: Sequence[str], repeat: int = 1) -> Dict[str, Any]:
    """Tekrarlayan blok (ornek: "x12 Encoder Blocks")."""
    return {"label": label, "layers": list(layer_ids), "repeat": int(repeat)}


def graph(
    name: str,
    layers: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    groups: Sequence[Dict[str, Any]] = (),
    **meta: Any,
) -> Dict[str, Any]:
    g: Dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "name": name,
        "layers": list(layers),
        "edges": list(edges),
        "groups": list(groups),
    }
    g.update(meta)
    validate(g)
    return g


def pack_f32(values: Sequence[float]) -> str:
    """float32 dizisini base64'e cevirir (little-endian)."""
    buf = struct.pack("<%df" % len(values), *values)
    return base64.b64encode(buf).decode("ascii")


def pack_array(arr: Any) -> str:
    """numpy dizisini base64 float32'ye cevirir; numpy yoksa listeye duser."""
    try:
        import numpy as np

        a = np.ascontiguousarray(np.asarray(arr, dtype="<f4").ravel())
        return base64.b64encode(a.tobytes()).decode("ascii")
    except ImportError:
        return pack_f32(list(arr))


def frame(
    step: int,
    activations: Dict[str, Any],
    *,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Tek bir zaman adimini tarif eder."""
    f: Dict[str, Any] = {
        "step": int(step),
        "act": {k: pack_array(v) for k, v in activations.items()},
    }
    if epoch is not None:
        f["epoch"] = int(epoch)
    if metrics:
        f["metrics"] = {k: float(v) for k, v in metrics.items()}
    return f


def validate(g: Dict[str, Any]) -> None:
    """Ciziciye gonderilmeden once tutarlilik kontrolu.

    Bozuk grafigi sessizce gondermek, tarayicida bos siyah ekranla sonuclanir
    ve hata ayiklamasi zordur; burada patlamasi cok daha iyi.
    """
    ids = [l["id"] for l in g["layers"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"katman id'leri benzersiz olmali, tekrar edenler: {sorted(dupes)}")
    known = set(ids)
    for e in g["edges"]:
        for side in ("from", "to"):
            if e[side] not in known:
                raise ValueError(f"kenar bilinmeyen katmana isaret ediyor: {e[side]!r}")
        if "weightsCount" in e and "shape" in e:
            want = e["shape"][0] * e["shape"][1]
            if e["weightsCount"] != want:
                raise ValueError(
                    f"{e['from']}->{e['to']}: agirlik sayisi {e['weightsCount']} "
                    f"ama shape {e['shape']} => {want} bekleniyordu"
                )
    for grp in g["groups"]:
        for lid in grp["layers"]:
            if lid not in known:
                raise ValueError(f"grup {grp['label']!r} bilinmeyen katman iceriyor: {lid!r}")
