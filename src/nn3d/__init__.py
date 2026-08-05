"""nn3d - sinir aglarini tarayicida canli 3D olarak izle.

Hizli baslangic (Keras):

    import nn3d

    view = nn3d.show(model, input_labels=ozellik_adlari)   # tarayici acilir
    model.fit(X, y, callbacks=[nn3d.Monitor(X[:1], every=20)])

Notebook'ta calisiyorsan hicbir sey yapman gerekmez; sunucu cekirdek acik
kaldigi surece yasar. Duz bir .py betiginde betik bitince surec de biter,
o yuzden sonunda `view.wait()` cagir.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from . import schema
from .server import Server

__version__ = "0.1.0"
__all__ = ["show", "Monitor", "View", "Server", "schema"]

_active: Optional["View"] = None


class View:
    """Acik bir gorsellestirme oturumu: sunucu + aktivasyon muslugu."""

    def __init__(
        self,
        model: Any,
        *,
        name: Optional[str] = None,
        input_labels: Optional[Sequence[str]] = None,
        output_labels: Optional[Sequence[str]] = None,
        max_neurons: int = schema.DEFAULT_MAX_NEURONS,
        port: int = 8092,
    ):
        from .keras_adapter import ActivationTap, build_graph

        self.model = model
        self.graph = build_graph(
            model,
            name=name,
            input_labels=input_labels,
            output_labels=output_labels,
            max_neurons=max_neurons,
        )
        self.tap = ActivationTap(model, max_neurons=max_neurons)
        self.server = Server(self.graph, port=port)
        self._step = 0

    @property
    def url(self) -> str:
        return self.server.url

    def update(
        self,
        x: Any,
        *,
        metrics: Optional[Dict[str, float]] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Bir ornegi ileri yayar ve o anki aktivasyonlari tarayiciya yollar."""
        acts = self.tap.read(x)
        self.server.push(schema.frame(self._step, acts, epoch=epoch, metrics=metrics))
        self._step += 1

    def open_browser(self) -> None:
        self.server.open_browser()

    def wait(self) -> None:
        self.server.wait()

    def close(self) -> None:
        self.server.close()


def show(
    model: Any,
    *,
    sample: Any = None,
    input_labels: Optional[Sequence[str]] = None,
    output_labels: Optional[Sequence[str]] = None,
    name: Optional[str] = None,
    max_neurons: int = schema.DEFAULT_MAX_NEURONS,
    port: int = 8092,
    open_browser: bool = True,
) -> View:
    """Modeli cizer ve tarayiciyi acar.

    sample verilirse ilk kare de hemen gonderilir, yani ekran bos kalmaz.
    """
    global _active
    if _active is not None:
        _active.close()
    view = View(
        model,
        name=name,
        input_labels=input_labels,
        output_labels=output_labels,
        max_neurons=max_neurons,
        port=port,
    )
    _active = view
    if sample is not None:
        view.update(sample)
    if open_browser:
        view.open_browser()
    print(f"[nn3d] {view.url}  ({view.graph['totalParams']:,} parametre)")
    return view


def wait() -> None:
    """Acik oturumu ayakta tutar (duz .py betikleri icin)."""
    if _active is None:
        raise RuntimeError("acik bir nn3d oturumu yok - once nn3d.show(model) cagir")
    _active.wait()


def __getattr__(name: str) -> Any:
    """nn3d.Monitor'a ilk erisimde keras'i tembel yukler (PEP 562).

    Boylece `import nn3d` Keras kurulu olmayan bir ortamda da calisir; sema ve
    sunucu baska cerceveler icin tek basina kullanilabilir kalir.
    """
    if name == "Monitor":
        from .monitor import Monitor

        return Monitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
