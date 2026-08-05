"""Sifir bagimlilikli yerel sunucu: statik dosyalar + canli aktivasyon akisi.

Neden WebSocket degil de SSE (Server-Sent Events)?
Akis tek yonlu: Python -> tarayici. SSE tam olarak bunun icin var, stdlib
http.server ile calisir, tarayicida `new EventSource(...)` kadar basit ve
baglanti koptugunda kendi kendine yeniden baglanir. WebSocket icin ya harici
bir paket gerekirdi ya da el yazmasi bir cerceve/handshake katmani -- ikisi de
"pip install nn3d yeter" hedefini bozardi.

Geri bildirim (hover, kamera) tarayicida kaliyor; Python'a donmesi gerekmiyor.
"""

from __future__ import annotations

import json
import queue
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

STATIC_DIR = Path(__file__).parent / "static"

# Tarayici cizmeye yetisemezse kuyruk buyumesin: egitim dongusu asla
# gorsellestirme yuzunden yavaslamamali ve ekranda ESKI degil EN SON durum
# gorunmeli. Bu yuzden kuyruk kucuk ve dolunca en eski kare atilir.
_QUEUE_SIZE = 8

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


class _QuietServer(ThreadingHTTPServer):
    """Baglanti kopmalarinda traceback basmayan HTTP sunucusu.

    Sekme kapandiginda ya da sunucu tarayici bagliyken durduruldugunda
    socketserver varsayilan olarak stderr'e koca bir traceback doker. Bu
    tamamen normal bir olay ve notebook ciktisinda hata sanilip panik
    yaratiyor. Gercek hatalar hala gorunur.
    """

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class _Client:
    """Tek bir tarayici sekmesi."""

    def __init__(self) -> None:
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=_QUEUE_SIZE)
        self.dropped = 0

    def put(self, payload: str) -> None:
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            try:
                self.q.get_nowait()      # en eskisini at
                self.dropped += 1
                self.q.put_nowait(payload)
            except queue.Empty:          # yarista baskasi aldiysa sorun degil
                pass


class Server:
    """Grafigi sunar ve kareleri bagli tum sekmelere yayinlar."""

    def __init__(self, graph: Dict[str, Any], *, port: int = 8092, host: str = "127.0.0.1"):
        self.graph = graph
        self.host = host
        self.status = "hazir"
        self._clients: List[_Client] = []
        self._lock = threading.Lock()
        self._last_frame: Optional[str] = None
        self.frames_sent = 0

        self.port = _free_port(host, port)
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass                      # egitim ciktisini kirletme

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/api/graph":
                    self._json(server.graph)
                elif path == "/api/stream":
                    server._stream(self)
                else:
                    self._static(path)

            # -- yardimcilar
            def _json(self, obj: Any) -> None:
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", _MIME[".json"])
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _static(self, path: str) -> None:
                rel = "index.html" if path in ("/", "") else path.lstrip("/")
                target = (STATIC_DIR / rel).resolve()
                # Dizin disina cikmayi engelle (../../etc/passwd)
                if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                    self.send_error(404)
                    return
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", _MIME.get(target.suffix, "application/octet-stream"))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self._httpd = _QuietServer((host, self.port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ akis
    def _stream(self, handler: BaseHTTPRequestHandler) -> None:
        client = _Client()
        with self._lock:
            self._clients.append(client)
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("Connection", "keep-alive")
            handler.send_header("X-Accel-Buffering", "no")
            handler.end_headers()

            # Yeni acilan sekme bos ekran gormesin: once yapiyi, sonra varsa
            # en son kareyi hemen yolla.
            self._write(handler, "graph", json.dumps(self.graph))
            if self._last_frame:
                self._write(handler, "frame", self._last_frame)

            while True:
                try:
                    payload = client.q.get(timeout=15)
                    self._write(handler, "frame", payload)
                except queue.Empty:
                    handler.wfile.write(b": ping\n\n")   # baglantiyi canli tut
                    handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                  # sekme kapandi, normal
        finally:
            with self._lock:
                if client in self._clients:
                    self._clients.remove(client)

    @staticmethod
    def _write(handler: BaseHTTPRequestHandler, event: str, data: str) -> None:
        handler.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
        handler.wfile.flush()

    # ------------------------------------------------------------------- API
    def push(self, frame: Dict[str, Any]) -> None:
        """Bir kareyi tum bagli sekmelere yayinlar. Asla bloklamaz."""
        payload = json.dumps(frame)
        self._last_frame = payload
        self.frames_sent += 1
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            c.put(payload)

    def set_graph(self, graph: Dict[str, Any]) -> None:
        self.graph = graph

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def open_browser(self) -> None:
        webbrowser.open(self.url)

    def wait(self) -> None:
        """Sekme acik kalsin diye sunucuyu ayakta tutar (Ctrl+C ile cikilir)."""
        print(f"[nn3d] {self.url} acik - kapatmak icin Ctrl+C")
        try:
            while True:
                self._thread.join(1.0)
        except KeyboardInterrupt:
            print("\n[nn3d] kapatiliyor")
            self.close()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _free_port(host: str, preferred: int) -> int:
    """Tercih edilen portu dene, gercekten doluysa isletim sisteminden bos bir
    port iste.

    SO_REUSEADDR sart: egitim betigini yeniden calistirdiginda onceki soket
    birkac dakika TIME_WAIT'te kaliyor. Bu secenek olmadan yoklama portu
    "dolu" saniyor, sunucu her seferinde baska bir porta kaciyor ve
    kullanicinin acik tarayici sekmesi oluyordu. ThreadingHTTPServer zaten
    allow_reuse_address kullaniyor, yoklamanin da ayni kurala uymasi gerek.
    """
    for candidate in (preferred, 0):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, candidate))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("bos port bulunamadi")
