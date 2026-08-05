"""Egitim sirasinda agi canli yayinlayan Keras callback'i.

Bu modul keras'i ICE AKTARIR. `import nn3d` bunu tetiklemez -- paketin
__init__.py'si `nn3d.Monitor`'a ilk erisildiginde tembel yukler. Boylece
sema/sunucu tarafi Keras kurulmamis ortamlarda da kullanilabilir kalir.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import keras

from . import schema


class Monitor(keras.callbacks.Callback):
    """Egitimi tarayiciya canli yayinlar.

        model.fit(X, y, callbacks=[nn3d.Monitor(X[:1], every=20)])

    `every` neden var: her batch'te ayrica bir ileri yayilim yapmak egitimi
    olcülebilir sekilde yavaslatir. 20 batch'te bir gonderim akici gorunur ve
    maliyeti ihmal edilebilir kalir.
    """

    def __init__(
        self,
        sample: Any,
        *,
        every: int = 20,
        input_labels: Optional[Sequence[str]] = None,
        output_labels: Optional[Sequence[str]] = None,
        max_neurons: int = schema.DEFAULT_MAX_NEURONS,
        port: int = 8092,
        open_browser: bool = True,
    ):
        super().__init__()
        self.sample = sample
        self.every = max(1, int(every))
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.max_neurons = max_neurons
        self.port = port
        self.open_browser = open_browser
        self.view: Optional[Any] = None
        self._epoch = 0
        self._batch = 0

    @property
    def url(self) -> Optional[str]:
        return self.view.url if self.view else None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        from . import show

        self.view = show(
            self.model,
            sample=self.sample,
            input_labels=self.input_labels,
            output_labels=self.output_labels,
            max_neurons=self.max_neurons,
            port=self.port,
            open_browser=self.open_browser,
        )
        self.view.server.status = "egitim"

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._epoch = epoch

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._batch += 1
        if self.view is None or self._batch % self.every:
            return
        self._send(self._epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # Epoch sonunda dogrulama metrikleri de geldigi icin her halukarda yolla.
        if self.view is not None:
            self._send(epoch, logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.view is not None:
            self.view.server.status = "egitim bitti"
            print(f"[nn3d] egitim bitti - {self.view.url} hala acik")

    def _send(self, epoch: int, logs: Optional[Dict[str, Any]]) -> None:
        assert self.view is not None
        metrics = {k: float(v) for k, v in (logs or {}).items() if _is_number(v)}
        self.view.update(self.sample, epoch=epoch, metrics=metrics)


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
