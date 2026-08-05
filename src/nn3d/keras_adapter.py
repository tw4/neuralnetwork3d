"""Keras 3 / TensorFlow modellerini nn3d semasina cevirir.

Iki is yapar:
  build_graph(model) -> agin yapisi (bir kez)
  ActivationTap(model).read(x) -> o anki aktivasyonlar (her adimda)

Tasarim notu -- SANALLASTIRMA:
Bir Dense(3072) katmanini 3072 nokta olarak cizmek hem okunmaz hem de onceki
768'lik katmanla arasinda 2.3 milyon kenar demektir; tarayici olur. Bu yuzden
her katmandan esit araliklarla `max_neurons` kadar TEMSILCI noron secilir ve
hem aktivasyonlar hem agirliklar ayni indislere gore kirpilir. Boylece ekranda
gordugun her nokta ve her cizgi GERCEK bir noron/agirliktir -- uydurma degil,
sadece bir alt kume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import schema

# Kendi agirligi olmayan, girisi aynen (ya da olcekleyerek) geciren katmanlar.
# Bunlar ince sutunlar olarak cizilir ve noron i -> noron i duz baglanir.
_PASSTHROUGH = {
    "Dropout", "SpatialDropout1D", "SpatialDropout2D", "GaussianDropout",
    "BatchNormalization", "LayerNormalization", "GroupNormalization",
    "Activation", "ReLU", "LeakyReLU", "ELU", "PReLU", "Softmax",
    "MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
    "AveragePooling1D", "AveragePooling2D", "AveragePooling3D",
    "GlobalMaxPooling2D", "GlobalAveragePooling2D",
    "ZeroPadding2D", "Cropping2D", "UpSampling2D",
}

_CONV = {"Conv1D", "Conv2D", "Conv3D", "SeparableConv2D", "DepthwiseConv2D"}
_DECONV = {"Conv2DTranspose", "Conv1DTranspose", "Conv3DTranspose"}
_RNN = {"LSTM", "GRU", "SimpleRNN", "Bidirectional", "ConvLSTM2D"}
_RESHAPE = {"Flatten", "Reshape", "Permute", "RepeatVector"}


def _kind(layer: Any) -> str:
    cls = type(layer).__name__
    if cls in _PASSTHROUGH:
        return "passthrough"
    if cls in _CONV or cls in _DECONV:
        return "conv"
    if cls in _RNN:
        return "rnn"
    if cls in _RESHAPE:
        return "reshape"
    return "dense"


def _activation_name(layer: Any) -> Optional[str]:
    """Kartin ucuncu satirinda kirmizi ile yazacak [ReLU] etiketi."""
    cls = type(layer).__name__
    if cls in ("ReLU", "LeakyReLU", "ELU", "PReLU", "Softmax"):
        return cls
    act = getattr(layer, "activation", None)
    if act is None:
        return None
    name = getattr(act, "__name__", None) or str(act)
    if name in ("linear", "None"):
        return None
    # relu -> ReLU, softmax -> Softmax, tanh -> Tanh
    special = {"relu": "ReLU", "elu": "ELU", "selu": "SELU", "gelu": "GELU",
               "prelu": "PReLU", "silu": "SiLU"}
    return special.get(name, name.capitalize())


def _out_shape(layer: Any) -> Tuple[Optional[int], ...]:
    try:
        return tuple(layer.output.shape)
    except Exception:
        return (None,)


def _display_size(layer: Any) -> int:
    """Kartta yazan ve nokta sayisini belirleyen 'genislik'.

    Her zaman SON eksen. Keras channels-last oldugu icin son eksen daima
    ozellik/kanal eksenidir; onceki eksenler uzamsal ya da zamansaldir ve
    kartta metin olarak gosterilir.

        Dense(256)              (None, 256)          -> 256
        Conv2D(32)              (None, 26, 26, 32)   ->  32
        BatchNorm (conv sonrasi)(None, 26, 26, 32)   ->  32
        LSTM(64, ret_seq=True)  (None, 60, 64)       ->  64
        Flatten                 (None, 1600)         -> 1600

    Onceki surumde yalnizca Conv katmanlarina bu kural uygulaniyordu; conv
    sonrasi BatchNorm/MaxPooling/LeakyReLU duzlestirilmis boyutu (21632)
    bildiriyor, bu da aktivasyon vektoru ile nokta sayisini uyumsuz
    birakiyordu.
    """
    shape = _out_shape(layer)
    if len(shape) <= 1:
        return 1
    return int(shape[-1] or 1)


def _shape_text(layer: Any) -> Optional[str]:
    shape = _out_shape(layer)
    dims = [s for s in shape[1:] if s is not None]
    if len(dims) >= 2:
        return "x".join(str(d) for d in dims)
    return None


def _input_shape(model: Any) -> Tuple[List[int], int]:
    """Giris katmaninin (boyutlar, cizilecek noron sayisi) ikilisi.

    Burada _display_size'in "son eksen" kurali BILEREK uygulanmiyor: bir
    goruntu girisi icin son eksen kanal sayisidir (28x28x1 -> 1) ve tek bir
    nokta cizmek anlamsiz olur. Giris ham veridir, ozellik degil; bu yuzden
    duzlestirip (784) sanallastiriyoruz. Gercek sekil kartta metin olarak
    zaten gorunuyor.
    """
    if not getattr(model, "inputs", None):
        return [], 1
    dims = [int(s) for s in tuple(model.inputs[0].shape)[1:] if s]
    if not dims:
        return [], 1
    return dims, int(np.prod(dims))


def _pick(n: int, k: int) -> np.ndarray:
    """n elemandan esit araliklarla k temsilci indis secer."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def _weight_matrix(layer: Any) -> Optional[np.ndarray]:
    """Katmanin agirligini (giris_birimi, cikis_birimi) matrisine indirger.

    Amac gorsel: iki noron arasindaki baglantinin isareti ve siddeti. Cok
    boyutlu cekirdekler bu iki boyuta ozetlenir.
    """
    ws = layer.get_weights()
    if not ws:
        return None
    k = np.asarray(ws[0], dtype=np.float32)
    cls = type(layer).__name__

    if k.ndim == 2:
        if cls in ("LSTM", "GRU"):
            # kernel: (giris, kapi_sayisi * birim). Kapilar uzerinden ortalama.
            gates = 4 if cls == "LSTM" else 3
            if k.shape[1] % gates == 0:
                units = k.shape[1] // gates
                return k.reshape(k.shape[0], gates, units).mean(axis=1)
        return k

    if k.ndim >= 3:
        # Konvolusyon: uzamsal eksenler uzerinde topla -> (giris_k, cikis_k)
        spatial = tuple(range(k.ndim - 2))
        m = k.sum(axis=spatial)
        if cls in _DECONV:
            # Conv2DTranspose cekirdegi (kh, kw, filtre, giris_kanali) seklinde
            m = m.T
        return m
    return None


def build_graph(
    model: Any,
    *,
    name: Optional[str] = None,
    input_labels: Optional[Sequence[str]] = None,
    output_labels: Optional[Sequence[str]] = None,
    max_neurons: int = schema.DEFAULT_MAX_NEURONS,
    max_edges_per_pair: int = 4096,
) -> Dict[str, Any]:
    """Keras modelini ciziciye gonderilecek grafige cevirir."""
    layers: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    picks: Dict[str, np.ndarray] = {}

    # --- giris katmani (Keras'ta Sequential'da gorunmez, kendimiz uyduruyoruz)
    in_dims, in_size = _input_shape(model)
    in_pick = _pick(in_size, max_neurons)
    picks["__input__"] = in_pick
    labels = None
    if input_labels is not None:
        lab = list(input_labels)
        labels = [lab[i] if i < len(lab) else f"in[{i}]" for i in in_pick]
    layers.append(schema.layer(
        "__input__", "Input", "input", in_size,
        shape_text="x".join(str(d) for d in in_dims) if len(in_dims) > 1 else None,
        neuron_labels=labels, max_neurons=max_neurons,
    ))

    prev_id = "__input__"
    prev_size = in_size
    keras_layers = list(model.layers)

    for i, kl in enumerate(keras_layers):
        kind = _kind(kl)
        size = _display_size(kl)
        lid = kl.name
        is_last = i == len(keras_layers) - 1

        sel = _pick(size, max_neurons)
        picks[lid] = sel

        nlabels = None
        if is_last and output_labels is not None:
            ol = list(output_labels)
            nlabels = [ol[j] if j < len(ol) else f"out[{j}]" for j in sel]

        layers.append(schema.layer(
            lid,
            _pretty_label(kl),
            "output" if is_last else kind,
            size,
            activation=_activation_name(kl),
            shape_text=_shape_text(kl),
            neuron_labels=nlabels,
            max_neurons=max_neurons,
            params=int(kl.count_params()),
        ))

        # --- kenar: onceki katmandan buraya
        W = _weight_matrix(kl)
        src_pick = picks[prev_id]
        if W is not None and W.shape[0] == prev_size:
            sub = W[np.ix_(src_pick, sel)]
            sub = _thin(sub, max_edges_per_pair)
            edges.append(schema.edge(
                prev_id, lid, kind="dense",
                shape=sub.shape, weights=sub.ravel(order="C"),
            ))
        elif kind == "passthrough" and size == prev_size:
            edges.append(schema.edge(prev_id, lid, kind="identity"))
        else:
            # Flatten/Reshape ya da boyutu tutmayan agirlik: yapisal baglanti
            edges.append(schema.edge(prev_id, lid, kind="reshape"))

        # RNN'in kendine donen baglantisi -- gorselde mor halkalar
        if kind == "rnn":
            edges.append(schema.edge(lid, lid, kind="recurrent"))

        prev_id, prev_size = lid, size

    return schema.graph(
        name or getattr(model, "name", None) or "model",
        layers, edges,
        framework="keras",
        totalParams=int(model.count_params()),
    )


# Uzun sinif adlari kartlari sisirip komsulariyla cakistiriyor. Kisaltmalar
# yaygin ve tanidik oldugu icin bilgi kaybi yok.
_SHORT = {
    "BatchNormalization": "BatchNorm",
    "LayerNormalization": "LayerNorm",
    "GroupNormalization": "GroupNorm",
    "MaxPooling1D": "MaxPool1D", "MaxPooling2D": "MaxPool2D", "MaxPooling3D": "MaxPool3D",
    "AveragePooling1D": "AvgPool1D", "AveragePooling2D": "AvgPool2D",
    "GlobalMaxPooling2D": "GlobalMaxPool",
    "GlobalAveragePooling2D": "GlobalAvgPool",
    "SpatialDropout2D": "SpDropout2D",
    "Conv2DTranspose": "Conv2DT", "Conv1DTranspose": "Conv1DT",
    "SeparableConv2D": "SepConv2D",
    "Bidirectional": "BiRNN",
}


def _pretty_label(layer: Any) -> str:
    """dense_3 -> "Dense 3" gibi okunakli, KISA bir baslik."""
    cls = type(layer).__name__
    short = _SHORT.get(cls, cls)
    raw = layer.name
    suffix = raw.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return f"{short} {suffix}"
    # Keras varsayilan adi sinif adinin snake_case hali ("batch_normalization").
    # Alt tireleri atmadan karsilastirinca hicbir zaman eslesmiyor ve kart
    # "BatchNormalization (batch_normalization)" gibi ise yaramaz sekilde
    # uzuyordu.
    if raw.replace("_", "").lower().startswith(cls.lower()):
        return short
    return f"{short} ({raw})"


def _thin(m: np.ndarray, budget: int) -> np.ndarray:
    """Kenar sayisi butceyi asiyorsa en zayif agirliklari sifirla.

    Matrisin sekli korunur (cizici sekle guveniyor); sadece degerler sifirlanir
    ve cizici sifir agirlikli kenari hic cizmez.
    """
    total = m.size
    if total <= budget:
        return m
    flat = np.abs(m).ravel()
    thresh = np.partition(flat, total - budget)[total - budget]
    out = m.copy()
    out[np.abs(out) < thresh] = 0.0
    return out


class ActivationTap:
    """Modelin ara katman ciktilarini okur.

    Egitim sirasinda her cagrida yeni bir Model kurmak cok pahali olurdu;
    probe modeli bir kez kurulur ve tekrar tekrar kullanilir.
    """

    def __init__(self, model: Any, *, max_neurons: int = schema.DEFAULT_MAX_NEURONS):
        import keras

        self.model = model
        self.max_neurons = max_neurons
        self.ids = ["__input__"] + [l.name for l in model.layers]
        self._picks: Dict[str, np.ndarray] = {}

        _, in_size = _input_shape(model)
        self._picks["__input__"] = _pick(in_size, max_neurons)
        for l in model.layers:
            self._picks[l.name] = _pick(_display_size(l), max_neurons)

        self._probe = keras.Model(
            inputs=model.inputs,
            outputs=[l.output for l in model.layers],
        )

    def read(self, x: Any) -> Dict[str, np.ndarray]:
        """Tek bir ornek (ya da batch) icin katman aktivasyonlarini dondurur."""
        x = np.asarray(x, dtype="float32")
        if x.shape[1:] != tuple(s for s in self.model.inputs[0].shape[1:]):
            x = x.reshape((-1,) + tuple(int(s) for s in self.model.inputs[0].shape[1:]))
        outs = self._probe.predict(x, verbose=0)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]

        acts: Dict[str, np.ndarray] = {}
        # Giris duzlestirilir (build_graph de oyle sayiyor), ortalanmaz:
        # ham piksel/ozellik degerlerini oldugu gibi gormek istiyoruz.
        flat_in = x[0].ravel()
        sel_in = self._picks["__input__"]
        acts["__input__"] = flat_in[sel_in[sel_in < flat_in.shape[0]]].astype("float32")
        for l, o in zip(self.model.layers, outs):
            acts[l.name] = self._reduce(np.asarray(o), self._picks[l.name])
        return acts

    @staticmethod
    def _reduce(a: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """(batch, ...) tensorunu secili noronlarin tek boyutlu vektorune indirir.

        Conv ciktilarinda uzamsal eksenler ortalanir -> kanal basina tek deger,
        yani ekrandaki nokta "bu ozellik haritasi ne kadar aktif" demek olur.
        """
        a = a[0] if a.ndim > 1 else a           # batch'in ilk ornegi
        if a.ndim > 1:
            a = a.mean(axis=tuple(range(a.ndim - 1)))
        sel = sel[sel < a.shape[0]]
        return a[sel].astype("float32")
