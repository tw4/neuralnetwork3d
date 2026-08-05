# nn3d — Geliştirici Dokümantasyonu

Bu belge kütüphanenin içini anlatır: modüller arası veri akışı, wire protokolü,
yeni çerçeve adaptörü yazma rehberi ve görüntüleyici geliştirme ortamı.

---

## İçindekiler

1. [Genel Mimari](#genel-mimari)
2. [Modüller](#modüller)
   - [schema.py — Wire Protokolü](#schempy--wire-protokolü)
   - [server.py — HTTP + SSE Sunucusu](#serverpy--http--sse-sunucusu)
   - [keras_adapter.py — Keras Entegrasyonu](#keras_adapterpy--keras-entegrasyonu)
   - [monitor.py — Keras Callback](#monitorpy--keras-callback)
   - [\_\_init\_\_.py — Genel API](#__init__py--genel-api)
3. [Veri Akışı](#veri-akışı)
4. [Wire Protokolü Referansı](#wire-protokolü-referansı)
5. [Yeni Çerçeve Adaptörü Yazmak](#yeni-çerçeve-adaptörü-yazmak)
6. [Görüntüleyici Geliştirme](#görüntüleyici-geliştirme)
7. [Tasarım Kararları](#tasarım-kararları)

---

## Genel Mimari

```
┌─────────────────────────────────────────────────────┐
│                   Python tarafı                      │
│                                                     │
│  keras_adapter.build_graph(model)                   │
│         │ graph dict (tek seferlik)                 │
│         ▼                                           │
│      Server ──── /api/graph ──────────────────────┐ │
│         │                                          │ │
│  ActivationTap.read(x)                             │ │
│         │ {layer_id: float32[]} (her adımda)       │ │
│         ▼                                          │ │
│   schema.frame(step, acts) ─── /api/stream (SSE) ─┤ │
│                                                    │ │
└────────────────────────────────────────────────────┘ │
                                                       │
                          HTTP / SSE                   │
                                                       │
┌──────────────────────────────────────────────────────┘
│                 Tarayıcı (Three.js)
│
│  EventSource("/api/stream")
│    ├── event: graph  → ağ topolojisi çizilir (bir kez)
│    └── event: frame  → nöron renkleri güncellenir (her adımda)
│
└──────────────────────────────────────────────────────
```

İki bağımsız veri yolu var:

- **Topoloji** (`graph`): modelin katmanları, kenarları, grupları. Eğitim
  başlamadan önce bir kez gönderilir, geometri hiç değişmez.
- **Aktivasyonlar** (`frame`): her `update()` çağrısında gönderilir, sadece
  renk tamponlarını günceller.

---

## Modüller

### schema.py — Wire Protokolü

Python ile tarayıcı arasındaki **tek sözleşme**. Tarayıcı Keras'ı bilmez;
sadece bu şemayı bilir. Bu sayede aynı görüntüleyici farklı çerçeveleri
destekleyebilir.

#### Sabitler

| Sabit | Değer | Açıklama |
|---|---|---|
| `SCHEMA_VERSION` | `1` | Geriye uyumluluğu izlemek için |
| `DEFAULT_MAX_NEURONS` | `16` | Katman başına çizilen nokta sayısı |
| `KINDS` | `tuple[str]` | Geçerli katman türleri |

#### Katman Türleri (`KINDS`)

| Tür | Ne zaman kullanılır |
|---|---|
| `input` | Modelin giriş katmanı; sol tarafta özellik etiketleriyle |
| `dense` | Tam bağlı katman |
| `conv` | Konvolüsyon; kart üzerinde uzamsal boyut yazar |
| `rnn` | LSTM/GRU; kendine dönen kavis çizilir |
| `reshape` | Flatten/Reshape; ağırlık yok, şekil değişir |
| `passthrough` | Dropout/BatchNorm; ince sütun, nöron i→i doğrudan bağlanır |
| `output` | Son katman; sağ tarafta sınıf etiketleriyle |

#### Fonksiyonlar

```python
schema.layer(id, label, kind, size, *, activation=None, shape_text=None,
             neuron_labels=None, max_neurons=16, params=0) -> dict
```

Tek bir katmanı tarif eder. `size` gerçek nöron sayısıdır (kart üzerinde yazar);
`max_neurons` kaç tanesinin çizileceğidir.

---

```python
schema.edge(src, dst, *, kind="dense", shape=None, weights=None) -> dict
```

| `kind` | Anlamı |
|---|---|
| `"dense"` | Ağırlıklı tam bağlantı |
| `"identity"` | Dropout/BatchNorm; nöron i→i düz çizgi |
| `"reshape"` | Flatten/Reshape; yapısal bağlantı, ağırlık yok |
| `"recurrent"` | RNN'in kendine dönen halkası |

`weights` verildiğinde `pack_f32()` ile base64'e çevrilmiş row-major float32
dizisidir. Matris boyutu `shape[0] * shape[1]` olmalıdır (`validate()` bunu
kontrol eder).

---

```python
schema.group(label, layer_ids, repeat=1) -> dict
```

Tekrarlayan blok tanımı (örn. `"Encoder x6"`). `layers` içindeki tüm id'ler
mevcut katmanlara işaret etmeli; `validate()` bunu denetler.

---

```python
schema.graph(name, layers, edges, groups=(), **meta) -> dict
```

Topoloji dict'ini oluşturur ve `validate()` ile tutarlılığını kontrol eder.
`**meta` ile `framework`, `totalParams` gibi ek alanlar eklenir; bunlar
tarayıcının bilgi çubuğunda görünür.

---

```python
schema.frame(step, activations, *, epoch=None, metrics=None) -> dict
```

`activations`: `{layer_id: numpy_array}` eşlemesi. Değerler `pack_array()` ile
base64 float32'ye dönüştürülür.

---

```python
schema.pack_f32(values: list[float]) -> str   # Python listesinden
schema.pack_array(arr: Any) -> str            # numpy dizisinden (daha hızlı)
```

Her iki fonksiyon da little-endian float32 + base64 üretir. Tarayıcı tarafında:

```ts
const buf = Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer;
const floats = new Float32Array(buf);
```

---

```python
schema.validate(g: dict) -> None
```

`graph()` içinden otomatik çağrılır. Kontroller:
- Katman id'leri benzersiz olmalı
- Kenarlar bilinen katman id'lerine işaret etmeli
- `weightsCount == shape[0] * shape[1]` olmalı
- Grup içindeki katman id'leri mevcut olmalı

---

### server.py — HTTP + SSE Sunucusu

Sıfır bağımlılıklı yerel HTTP sunucusu. `ThreadingHTTPServer` + `BaseHTTPRequestHandler`
ile çalışır, harici paket gerektirmez.

#### `_QuietServer`

`ThreadingHTTPServer` alt sınıfı. Sekme kapanınca ya da sunucu tarayıcı bağlıyken
durdurulunca oluşan `BrokenPipeError` / `ConnectionResetError` gibi normal
bağlantı kopması hatalarını yutarak notebook çıktısına gereksiz traceback
basılmasını önler. Gerçek hatalar (`handle_error`) hâlâ görünür.

#### Rotalar

| Rota | Yanıt |
|---|---|
| `GET /` | `static/index.html` |
| `GET /api/graph` | `Content-Type: application/json`, topoloji (tek seferlik) |
| `GET /api/stream` | `Content-Type: text/event-stream`, SSE akışı |
| `GET /<dosya>` | `static/` altındaki statik dosya |

Statik dosya sunucusu `../../etc/passwd` gibi dizin çıkış girişimlerini 404 ile
reddeder.

#### SSE Olayları

```
event: graph
data: {"version":1,"name":"churn","layers":[...],"edges":[...],"groups":[...]}

event: frame
data: {"step":42,"epoch":1,"metrics":{"loss":0.31},"act":{"dense_1":"<b64>"}}

: ping
```

15 saniyelik kuyruk boşluğunda `": ping\n\n"` gönderilir; bu SSE yorumudur,
tarayıcı yoksayar ama bağlantıyı canlı tutar.

`X-Accel-Buffering: no` başlığı nginx gibi ters proxy'lerin SSE'yi arabelleğe
almasını engeller.

#### `_Client` — Kuyruk Stratejisi

Her bağlı sekme için ayrı `queue.Queue(maxsize=8)` tutulur. Kuyruk dolduğunda
**en eski kare atılır**, en yeni eklenir. Böylece:

- Eğitim döngüsü asla görselleştirme yüzünden bloklanmaz.
- Ekranda her zaman en güncel durum görünür.

`_Client.dropped` sayacı atlanan kare sayısını tutar (şu an yalnızca dahili).

#### `Server` API

| Üye | Açıklama |
|---|---|
| `push(frame_dict)` | Kareyi tüm sekmelere yayar, asla bloklamaz |
| `set_graph(graph)` | Sunucunun topoloji referansını günceller |
| `open_browser()` | `webbrowser.open(url)` |
| `wait()` | Ctrl+C bekler (duz `.py` betikleri için) |
| `close()` | HTTP sunucusunu durdurur |
| `url` | `http://127.0.0.1:<port>/` |
| `clients` | Anlık bağlı sekme sayısı (thread-safe) |
| `frames_sent` | Toplam yayınlanan kare sayısı |
| `status` | Durum metni; başlangıç: `"hazir"`, Monitor tarafından güncellenir |

#### `_free_port(host, preferred)`

`SO_REUSEADDR` ile tercih edilen portu dener. Betik yeniden çalıştırıldığında
önceki soket birkaç dakika `TIME_WAIT`'te kalır; bu seçenek olmadan port "dolu"
sanılır ve sunucu her seferinde farklı porta geçer. İstenen port doluysa OS'tan
rastgele boş port (`candidate=0`) ister.

---

### keras_adapter.py — Keras Entegrasyonu

İki işi var:

1. `build_graph(model)` → ağın yapısı (**bir kez** çalışır)
2. `ActivationTap(model).read(x)` → o anki aktivasyonlar (**her adımda**)

#### Katman Türü Tespiti

```python
_PASSTHROUGH = {"Dropout", "BatchNormalization", "LayerNormalization", ...}
_CONV        = {"Conv1D", "Conv2D", "Conv3D", "SeparableConv2D", ...}
_DECONV      = {"Conv2DTranspose", ...}
_RNN         = {"LSTM", "GRU", "SimpleRNN", "Bidirectional", ...}
_RESHAPE     = {"Flatten", "Reshape", "Permute", ...}
```

`_kind(layer)` bu kümelere bakarak tür döndürür; hiçbirine girmeyenler `"dense"`
sayılır.

#### `_activation_name(layer)` — Kart Etiketi

Katmanın aktivasyon fonksiyonunu kart üzerinde kırmızı `[ReLU]` olarak yazan
etiketi üretir. Hem `layer.activation` niteliğini hem de `ReLU`, `Softmax` gibi
bağımsız aktivasyon katmanlarını tanır. `"linear"` için `None` döner.

#### `_pretty_label(layer)` — Okunabilir Başlık

`dense_3` → `"Dense 3"` gibi kısa ve okunabilir kart başlığı üretir.
`_SHORT` sözlüğü uzun sınıf adlarını kısaltır:

```python
_SHORT = {
    "BatchNormalization": "BatchNorm",
    "Conv2DTranspose": "Conv2DT",
    "GlobalAveragePooling2D": "GlobalAvgPool",
    ...
}
```

Keras'ın ürettiği varsayılan isimler (`batch_normalization`) alt tire olmadan
sınıf adıyla karşılaştırılır; aksi hâlde `"BatchNormalization (batch_normalization)"`
gibi gereksiz uzun başlıklar çıkıyordu.

#### `_shape_text(layer)` — Boyut Metni

Çıkış tensörünün uzamsal eksenlerini `"28x28x32"` formatında döndürür; yalnızca
çok boyutlu çıkışlarda geçerlidir. Kart üzerinde yazılır.

#### `_display_size(layer)` — Kaç Nokta Çizilecek

Her zaman **son eksen**. Keras channels-last olduğu için son eksen daima
özellik/kanal eksendir; önceki eksenler uzamsal ya da zamansal.

```
Dense(256)                 (None, 256)       → 256
Conv2D(32, ...)            (None, 26, 26, 32) →  32
BatchNorm (conv sonrası)   (None, 26, 26, 32) →  32   ← önceki sürümde hatalıydı
LSTM(64, return_seq=True)  (None, 60, 64)    →  64
Flatten                    (None, 1600)      → 1600
```

**İstisna — giriş katmanı:** `_input_shape()` son eksen kuralını **kasıtlı
uygulamaz**. Görüntü girişinde son eksen kanal sayısıdır (`28×28×1 → 1`); tek
nokta anlamsız olur, bu yüzden giriş düzleştirilir (`784`).

#### `_pick(n, k)` — Sanal Nöron Seçimi

```python
np.unique(np.linspace(0, n-1, k).round().astype(int))
```

`n` elemandan eşit aralıklarla `k` temsilci indis seçer. `n ≤ k` ise hepsi
alınır. Aynı indisler hem ağırlık matrisini (satır + sütun) hem aktivasyon
vektörünü kesmek için kullanılır; böylece ekrandaki her nokta ve çizgi gerçek
bir nöron/ağırlıktır.

#### `_weight_matrix(layer)` — Görsel Ağırlık Matrisi

Hedef: iki nöron arasındaki bağlantının işareti ve şiddeti. Çok boyutlu
çekirdekler `(giriş_birimi, çıkış_birimi)` matrisine indirgenir:

| Katman | Çekirdek şekli | İndirgeme |
|---|---|---|
| Dense | `(giriş, çıkış)` | doğrudan |
| LSTM | `(giriş, 4*birim)` | kapılar üzerinden ortalama → `(giriş, birim)` |
| GRU | `(giriş, 3*birim)` | kapılar üzerinden ortalama → `(giriş, birim)` |
| Conv2D | `(kh, kw, giriş_k, çıkış_k)` | uzamsal eksenler toplanır → `(giriş_k, çıkış_k)` |
| Conv2DTranspose | `(kh, kw, filtre, giriş_k)` | uzamsal toplanır + **transpoze** |

#### `_thin(matrix, budget)` — Kenar Budama

`max_edges_per_pair=4096` aşılırsa en zayıf ağırlıklar sıfırlanır. Matrisin
şekli korunur (görüntüleyici şekle güveniyor); tarayıcı sıfır ağırlıklı kenarı
çizmez.

#### `build_graph` imzası

```python
build_graph(model, *, name=None, input_labels=None, output_labels=None,
            max_neurons=16, max_edges_per_pair=4096) -> dict
```

`input_labels` ve `output_labels`: giriş/çıkış katmanları için nöron etiketleri.
İndis sayısından az etiket verilirse geri kalanlar `"in[i]"` / `"out[i]"` olur.

#### `ActivationTap`

Keras'ın `Model(inputs=..., outputs=[her_katmanın_çıktısı])` özelliğini kullanır;
probe modeli **bir kez** kurulur ve her `read()` çağrısında `predict()` çalıştırılır.

`_reduce(a, sel)`:
- Batch'in ilk örneği alınır (`a[0]`).
- Çok boyutlu tensörlerde (conv) uzamsal eksenler **ortalanır** → kanal başına
  tek değer. Yani ekrandaki nokta "bu özellik haritası ne kadar aktif" demektir.
- Seçili indisler uygulanır.

Giriş aktivasyonları ise `_reduce` yerine düzleştirme (`ravel`) uygulanır:
ham piksel/özellik değerlerini olduğu gibi görmek istiyoruz.

---

### monitor.py — Keras Callback

`keras.callbacks.Callback` alt sınıfı. Tembel import: `import nn3d` bu dosyayı
yüklemez; `nn3d.Monitor`'a ilk erişimde `__init__.py`'deki `__getattr__` tetikler.
Keras kurulu olmayan ortamlarda paket yüklenebilir kalır.

#### Yapıcı imzası

```python
Monitor(sample, *, every=20, input_labels=None, output_labels=None,
        max_neurons=16, port=8092, open_browser=True)
```

| Parametre | Açıklama |
|---|---|
| `sample` | Aktivasyon okumak için kullanılan tek örnek |
| `every` | Kaç batch'te bir kare gönderilir |
| `input_labels` | Giriş nöronu etiketleri |
| `output_labels` | Çıkış nöronu etiketleri |
| `open_browser` | Eğitim başlayınca tarayıcı otomatik açılsın mı |

#### Callback Kancaları

| Kanca | Ne yapar |
|---|---|
| `on_train_begin` | `nn3d.show()` çağırır, sunucuyu başlatır |
| `on_epoch_begin` | Epoch numarasını saklar |
| `on_train_batch_end` | `self._batch % every == 0` ise kare gönderir |
| `on_epoch_end` | Her epoch sonunda (doğrulama metrikleri dahil) kare gönderir |
| `on_train_end` | Durum metnini `"egitim bitti"` yapar |

#### `url` özelliği

`monitor.url` — sunucu başladıktan sonra `http://127.0.0.1:<port>/` döner,
öncesinde `None`.

`every` parametresi: her batch'te ekstra ileri yayılım eğitimi yavaşlatır.
20'de bir akıcı görünür ve maliyeti ihmal edilebilir.

---

### \_\_init\_\_.py — Genel API

#### `show(model, ...) -> View`

```python
show(model, *, sample=None, input_labels=None, output_labels=None,
     name=None, max_neurons=16, port=8092, open_browser=True) -> View
```

- Önceki aktif `View` varsa önce kapatır (`_active.close()`).
- `build_graph` + `ActivationTap` + `Server` oluşturur.
- `sample` verilirse `view.update(sample)` ile ilk kareyi hemen gönderir
  (ekran boş kalmaz).
- `open_browser=False` verilirse tarayıcı otomatik açılmaz (notebook'larda
  ya da uzak sunucularda kullanışlı).
- `print(f"[nn3d] {url}  ({totalParams:,} parametre)")` yazar.

#### `wait()`

```python
nn3d.wait()
```

Aktif `View` yoksa `RuntimeError` fırlatır. Duz `.py` betiklerinde son satır
olarak kullanılır; Ctrl+C ile çıkılana kadar sunucuyu ayakta tutar.

#### `View`

| Metot / Özellik | Açıklama |
|---|---|
| `update(x, metrics=None, epoch=None)` | Tek örnek ileri yayar, kareyi tarayıcıya gönderir |
| `open_browser()` | `webbrowser.open(url)` |
| `wait()` | Ctrl+C bekler (duz `.py` betikleri için) |
| `close()` | HTTP sunucusunu durdurur |
| `url` | `http://127.0.0.1:<port>/` |
| `server` | Alttaki `Server` nesnesine erişim |
| `graph` | Build edilmiş topoloji dict'i |

#### Tembel `Monitor` import'u

`nn3d.Monitor`'a ilk erişimde `__getattr__` (PEP 562) tetikler ve
`monitor.py`'yi yükler. Böylece `import nn3d` Keras kurulu olmayan ortamlarda
da çalışır.

---

## Veri Akışı

### İlk bağlantı (sekme açıldığında)

```
Tarayıcı  GET /api/stream
Python    HTTP 200, Content-Type: text/event-stream
Python    event: graph   data: <topoloji JSON>
Python    event: frame   data: <son kaydedilen kare>   ← _last_frame, boş değilse
Tarayıcı  ağı çizer, renkleri uygular
```

Sonraki açılan sekmeler de aynı akışı alır; eğitim ortasında açılan sekme hemen
son durumu görür.

### Eğitim adımı

```
Python    view.update(X_sample, epoch=e, metrics={"loss": l})
             └─ ActivationTap.read(X_sample) → {layer_id: float32[]}
             └─ schema.frame(step, acts, epoch=e, metrics=...)
             └─ Server.push(frame_dict)
                    └─ json.dumps → _last_frame güncel
                    └─ tüm _Client.put(payload)   ← asla bloklamaz
```

### Sunucu durumu

`Server.status` string'i `Monitor` tarafından güncellenir
(`"egitim"`, `"egitim bitti"`); başlangıç değeri `"hazir"`. Tarayıcı bu alanı
durum çubuğunda gösterir.

---

## Wire Protokolü Referansı

### GRAPH mesajı (SSE event: graph)

```jsonc
{
  "version": 1,
  "name": "churn",
  "framework": "keras",
  "totalParams": 1601,
  "layers": [
    {
      "id": "__input__",
      "label": "Input",
      "kind": "input",
      "size": 16,            // gerçek nöron sayısı
      "params": 0,
      "display": { "maxNeurons": 16 },
      "neuronLabels": ["Tenure", "MonthlyCharges", ...],  // opsiyonel
      "shapeText": "16x16"   // çok boyutluysa, opsiyonel
    },
    {
      "id": "dense",
      "label": "Dense",
      "kind": "dense",
      "size": 32,
      "params": 544,
      "activation": "ReLU",
      "display": { "maxNeurons": 16 }
    }
    // ...
  ],
  "edges": [
    {
      "from": "__input__",
      "to": "dense",
      "kind": "dense",
      "shape": [16, 16],       // [src_pick_count, dst_pick_count]
      "weights": "<base64 f32>",
      "weightsCount": 256
    },
    {
      "from": "dropout",
      "to": "dense_1",
      "kind": "identity"       // ağırlık yok
    },
    {
      "from": "lstm",
      "to": "lstm",
      "kind": "recurrent"      // kendine dönen halka
    }
  ],
  "groups": [
    {
      "label": "Encoder x6",
      "layers": ["attn_1", "ffn_1"],
      "repeat": 6
    }
  ]
}
```

### FRAME mesajı (SSE event: frame)

```jsonc
{
  "step": 42,
  "epoch": 1,
  "metrics": {
    "loss": 0.3142,
    "accuracy": 0.8801,
    "val_loss": 0.3405,
    "val_accuracy": 0.8612
  },
  "act": {
    "__input__": "<base64 little-endian float32>",
    "dense":     "<base64 little-endian float32>",
    "dense_1":   "<base64 little-endian float32>"
  }
}
```

Her `act` değeri `display.maxNeurons` uzunluğunda (ya da daha kısa) bir float32
dizisidir. Tarayıcı `Float32Array` ile okur; değerlerin sırası `build_graph()`
içindeki `_pick()` ile belirlenen indislere karşılık gelir.

---

## Yeni Çerçeve Adaptörü Yazmak

Tarayıcı yalnızca `schema.py`'yi bilir. PyTorch, Unity veya başka bir çerçeve
için yeni bir adaptör yazmak şu iki fonksiyonu sağlamakla başlar:

### 1. `build_graph(model) -> dict`

Çerçeveye özgü model nesnesini `schema.graph()` dict'ine dönüştürün.

```python
# src/nn3d/pytorch_adapter.py  (örnek iskelet)
from . import schema
import numpy as np

def build_graph(model, *, name=None, max_neurons=16):
    layers = []
    edges  = []
    prev_id = "__input__"

    # --- giriş katmanını kendiniz oluşturun ---
    in_size = _infer_input_size(model)
    layers.append(schema.layer("__input__", "Input", "input", in_size,
                               max_neurons=max_neurons))

    for module in model.children():
        lid  = module.__class__.__name__.lower()
        kind = _kind(module)      # kendi eşlemenizi yazın
        size = _out_size(module)  # modülün çıkış boyutu
        params = sum(p.numel() for p in module.parameters())

        layers.append(schema.layer(lid, module.__class__.__name__, kind, size,
                                   max_neurons=max_neurons, params=params))

        W = _weight_matrix(module)  # (giriş, çıkış) matrisi ya da None
        if W is not None:
            src_pick = _pick(prev_size, max_neurons)
            dst_pick = _pick(size, max_neurons)
            sub = W[np.ix_(src_pick, dst_pick)]
            edges.append(schema.edge(prev_id, lid, kind="dense",
                                     shape=sub.shape, weights=sub.ravel()))
        else:
            edges.append(schema.edge(prev_id, lid, kind="reshape"))

        prev_id, prev_size = lid, size

    return schema.graph(name or "model", layers, edges, framework="pytorch",
                        totalParams=sum(p.numel() for p in model.parameters()))
```

### 2. `ActivationTap` muadili

```python
class ActivationTap:
    def __init__(self, model, *, max_neurons=16):
        self._acts = {}
        self.max_neurons = max_neurons
        self.model = model
        for name, module in model.named_modules():
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._acts[name] = out.detach().cpu().numpy()
        return hook

    def read(self, x) -> dict:
        self._acts.clear()
        with torch.no_grad():
            self.model(x)
        return {k: _reduce(v, _pick(v.shape[-1], self.max_neurons))
                for k, v in self._acts.items()}
```

### 3. Bağlama

```python
# src/nn3d/__init__.py içinde ya da yeni bir show_torch() fonksiyonunda
from .pytorch_adapter import build_graph, ActivationTap
view = View.__new__(View)
view.graph = build_graph(model)
view.tap   = ActivationTap(model)
view.server = Server(view.graph, port=port)
```

Tarayıcı tarafında **hiçbir değişiklik gerekmez**.

---

## Görüntüleyici Geliştirme

Derleme çıktısı `src/nn3d/static/index.html` içinde tek dosya olarak duruyor ve
wheel'e giriyor. `.gitignore` bu dosyayı kasıtlı olarak hariç tutmaz.

### Klasör yapısı

```
viewer/
├── index.html          # Vite giriş noktası (sadece <div id="root"> ve CSS)
├── package.json
├── vite.config.ts      # çıktı: ../src/nn3d/static/index.html
├── tsconfig.json
└── src/
    ├── main.tsx        # React root (createRoot)
    ├── App.tsx         # Uygulama state'i, SSE hookup, bileşen ağacı
    ├── ThreeScene.tsx  # Three.js canvas (useEffect + forwardRef)
    ├── Hud.tsx         # HUD overlay (meta, metrikler, durum, tooltip)
    ├── useSSE.ts       # SSE bağlantı hook'u
    ├── cards.ts        # Katman kartları (canvas dokusu üzerinde metin)
    ├── edges.ts        # Bézier kenar geometrisi ve boyama
    ├── layout.ts       # Katman konumlandırma algoritması
    ├── neurons.ts      # Nöron nokta bulutu (Points geometrisi)
    ├── theme.ts        # Renk sabitleri (arka plan, pozitif, negatif, vurgular)
    └── types.ts        # Graph / Layer / Frame TypeScript tipleri + unpackF32
```

Viewer React 18 + TypeScript ile yazılmıştır; `vite-plugin-singlefile` ile tek
`index.html` dosyasına derlenir.

### Bileşen mimarisi

```
App
├── ThreeScene (forwardRef → SceneHandle)   ← Three.js + imperative
└── Hud                                     ← React state → JSX
```

**`App.tsx`** uygulama state'ini tutar (`status`, `meta`, `metrics`) ve SSE
olaylarını bileşenlere dağıtır.

**`ThreeScene.tsx`** Three.js sahnesini `useEffect` içinde kurar; dışarıya iki
metodluk bir `SceneHandle` handle'ı expose eder:
- `applyGraph(g)` → topoloji değiştiğinde yeni geometri inşa eder
- `applyFrame(f)` → aktivasyon tamponunu günceller, dirty flag set eder

Tooltip (hover), 60 fps'de değiştiği için React state yerine `tipRef`
aracılığıyla imperativ olarak güncellenir.

**`Hud.tsx`** saf bir React bileşenidir: `status`, `meta`, `metrics` prop'ları
değişince yeniden render edilir. `tipRef` ile tooltip `<div>`'ini paylaşır.

**`useSSE.ts`** `EventSource` kurulumunu bir hook içinde soyutlar; callback
referansları `useRef` ile stabilize edilir, SSE bağlantısı yalnızca mount'ta
kurulur.

### Modüller

| Modül | Sorumluluğu |
|---|---|
| `main.tsx` | `createRoot` ile React'ı başlatır |
| `App.tsx` | State yönetimi, SSE hookup, `SceneHandle` ref'i |
| `ThreeScene.tsx` | Renderer, kamera, OrbitControls, hover/raycasting, render döngüsü |
| `Hud.tsx` | Meta panel, metrikler, durum çubuğu, tooltip div'i |
| `useSSE.ts` | `EventSource` hook'u |
| `layout.ts` | `computeLayout(graph)` → her katmanın 3D konumu ve nöron koordinatları |
| `neurons.ts` | `NeuronField`: tüm nöronlar tek `Points` nesnesinde; aktivasyona göre renk |
| `edges.ts` | `EdgeField`: ağırlık matrisinden Bézier eğrileri; renk = işaret, parlaklık = şiddet |
| `cards.ts` | `Cards`: her katman için katman adı / boyut / aktivasyon etiketli 2D kart |
| `theme.ts` | `THEME` sabiti — `bg`, `fog`, `pos` (turkuaz), `neg` (kırmızı), `dim` vb. |
| `types.ts` | `Graph`, `Layer`, `Edge`, `Frame` tipleri; `unpackF32(b64)` yardımcısı |

### Geliştirme ortamı

```bash
# Önce Python sunucusunu başlatın (SSE kaynağı)
PYTHONPATH=src python examples/churn_canli.py

# Ayrı terminalde Vite HMR sunucusu
cd viewer
npm install
npm run dev    # http://localhost:5173  (proxy: /api/* → :8092)
```

`vite.config.ts` içindeki `proxy` ayarı `/api/graph` ve `/api/stream`
isteklerini Python sunucusuna iletir; böylece HMR çalışırken gerçek verilerle
test edilir.

### Derleme ve commit

```bash
cd viewer
npm run build   # src/nn3d/static/index.html'i günceller
git add src/nn3d/static/index.html
git commit -m "viewer: <değişiklik>"
```

Derleme yapılmadan commit atılırsa kullanıcılar eski görüntüleyiciyi alır.

### Tarayıcı tarafı veri okuma

`useSSE.ts` hook'u SSE bağlantısını kapsüllemektedir:

```ts
// useSSE.ts içindeki EventSource kurulumu
const es = new EventSource("/api/stream");
es.addEventListener("graph", (e) =>
  onGraphRef.current(JSON.parse(e.data) as Graph)
);
es.addEventListener("frame", (e) =>
  onFrameRef.current(JSON.parse(e.data) as Frame)
);
```

`App.tsx`'ten kullanım:

```tsx
useSSE(handleGraph, handleFrame, handleError);

// handleGraph → sceneRef.current.applyGraph(g) + setMeta(...)
// handleFrame → sceneRef.current.applyFrame(f) + setMetrics(...) + setStatus(...)
```

### Klavye kısayolları

| Tuş | Eylem |
|---|---|
| `R` | Kamerayı sıfırla ve otomatik döndürmeyi yeniden başlat |
| Sürükleme | Otomatik döndürmeyi durdurur |

### Performans notları

- **Etiket boyama kısıtlaması:** `cards.refreshLabels()` her karede değil, ~7
  fps'de (140 ms aralığında) çağrılır. Canvas→GPU yükleme maliyetli; göz farkı
  ayırt etmiyor.
- **Hover raycaster eşiği:** `0.6` birim. Nöron aralığı 1.5 birim; 0.45 ile
  birkaç piksel sapmada hedef kaçıyordu, 0.6 komşuya taşmadan hedefi rahat tutar.
- **Additive blending:** Kenarlar `AdditiveBlending` ile çizilir; üst üste binen
  kenarlar parlar ve yoğunluk okunabilir hâle gelir.

---

## Tasarım Kararları

### WebSocket yerine SSE

Akış tek yönlü (Python → tarayıcı). SSE stdlib `http.server` ile çalışır,
tarayıcıda `new EventSource()` kadar basittir ve bağlantı koptuğunda kendi
kendine yeniden bağlanır. WebSocket ya harici bir paket ya da el yazması
handshake gerektirirdi; ikisi de `pip install nn3d yeter` hedefini bozardı.
Hover/kamera tarayıcıda kalıyor, Python'a dönmesi gerekmiyor.

### Sıfır zorunlu bağımlılık

`pyproject.toml`'da `dependencies = []`. Sunucu tamamen stdlib; numpy Keras ile
birlikte geldiği için ayrıca istenmez. `nn3d[keras]` ile kurulduğunda isteğe
bağlı bağımlılıklar (`keras>=3.0`, `numpy>=1.22`) eklenir.

### Tembel Keras import'u

`import nn3d` Keras'ı yüklemez. `nn3d.Monitor`'a ilk erişimde `__getattr__`
(PEP 562) tetikler. Böylece şema ve sunucu tarafı Keras kurulmayan ortamlarda
(örn. başka bir çerçeve adaptörü) da kullanılabilir.

### Katman sanallaştırma

`Dense(3072)` katmanını 3072 nokta çizmek 2.3 milyon kenar demektir; tarayıcı
kilitlenir. `_pick()` eşit aralıklı temsilci nöronlar seçer; aktivasyonlar ve
ağırlıklar aynı indislere göre kırpılır. Hover ipucu **gerçek nöron numarasını**
yazar (`nöron 1847 / 3072`) — hangi alt kümeye baktığını gizlemek yanıltıcı
olurdu.

### `_display_size` = son eksen

Keras channels-last olduğu için son eksen daima özellik/kanal eksendir. Bu kural
uygulanmadan önce Conv sonrası BatchNorm düzleştirilmiş boyutu (21632) bildiriyor,
aktivasyon vektörü ile nokta sayısı uyuşmuyordu. **Tek istisna:** giriş katmanı
(`_input_shape`) — görüntü girişinde son eksen kanal sayısıdır (`1`), tek nokta
anlamsız olur.

### `SO_REUSEADDR` zorunlu

Betik yeniden çalıştırıldığında önceki soket birkaç dakika `TIME_WAIT`'te kalır.
Bu seçenek olmadan `_free_port()` tercih edilen portu "dolu" sanır ve sunucu her
seferinde farklı porta geçer; kullanıcının açık sekme örülür.

### Kadraj — FOV formülü değil köşe izdüşümü

`fitToView()` sınırlayıcı kutunun köşelerini gerçekten ekrana izdüşürür ve taşma
oranını ölçer. FOV formülü kamerayı tam eksende varsayar; bizimki hafif eğik
durduğu için gerçek izdüşüm büyük çıkıp ağın sağını kırpıyordu. Kutu z
ekseninde %15'e daraltılır: derinliğin tamamı hesaba katılınca kameraya en yakın
köşe perspektifte şişiyor ve ağ ekranın yarısında kalıyordu; okunması gereken
her şey (sütunlar, kartlar, etiketler) zaten orta düzlemde. Pencere yeniden
boyutlandırılınca tekrar `fitToView()` çağrılır — kullanıcının elle yakınlaştırması
sıfırlanır, bu bilinçli bir tercih.

### `_QuietServer` — sessiz hata yönetimi

Sekme kapandığında ya da sunucu durdurulduğunda normal bağlantı kopmaları
(`BrokenPipeError`, `ConnectionResetError`) oluşur. Bunları `_QuietServer`
susturur; notebook çıktısında hata sanılan gürültü yaratmaz. Gerçek ağ hataları
hâlâ görünür.
