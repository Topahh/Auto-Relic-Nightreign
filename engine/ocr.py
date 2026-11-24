# engine/ocr.py
# OCR engine wrapper around PaddleOCR.
#
# PaddleOCR leaks host RAM (~25 MB per 1000 predict() calls, measured on
# 2026-11-24; malloc_trim does not help). A weeks-long session would saturate
# the RAM within hours. So the model runs in a separate WORKER PROCESS that is
# restarted between two batches once its memory passes [OCR] max_rss_mb.
# Only the OCR is restarted: the screen capture session (and its permission
# dialog) is untouched.
#
# The parent process never imports paddle: only the worker loads it.

import multiprocessing as mp
import os
import signal
import time
from typing import List, Optional

import numpy as np

from config.settings import Config


def _extract_texts(results) -> List[str]:
    texts: List[str] = []
    for result in results:
        if isinstance(result, dict) and 'rec_texts' in result:
            texts.extend(result['rec_texts'])                # PaddleOCR v3+
        elif isinstance(result, list):
            for line in result:                              # legacy format
                try:
                    texts.append(line[1][0])
                except Exception:
                    pass
    return texts


def _worker_main(conn, lang: str, doc_preprocess: bool) -> None:
    """Child process: load PaddleOCR once, then answer recognition requests."""
    # Ctrl+C in the terminal reaches the whole process group: the parent
    # stops the worker itself, the worker must not die mid-request.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import paddle
        from paddleocr import PaddleOCR
        try:
            has_gpu = (paddle.device.is_compiled_with_cuda()
                       and paddle.device.cuda.device_count() > 0)
        except Exception:
            has_gpu = None
        # Document orientation + UVDoc unwarping are made for photos of paper.
        # On a flat game screen they double the OCR time and can distort text.
        engine = PaddleOCR(
            lang=lang,
            use_textline_orientation=False,
            use_doc_orientation_classify=doc_preprocess,
            use_doc_unwarping=doc_preprocess,
        )
        conn.send(("READY", has_gpu))
    except Exception as e:
        conn.send(("FAIL", repr(e)))
        return

    while True:
        try:
            image = conn.recv()
        except (EOFError, OSError):
            return                      # parent is gone
        if image is None:
            return                      # clean stop
        try:
            conn.send(("OK", _extract_texts(engine.predict(image))))
        except Exception as e:
            conn.send(("ERR", repr(e)))


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


class OCREngine:
    """PaddleOCR running in a restartable worker process."""

    REQUEST_TIMEOUT = 30.0     # s, one recognition
    STARTUP_TIMEOUT = 180.0    # s, model loading

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self.restarts = 0
        print("[INIT] Loading OCR engine...")
        print(f"[INIT] OCR language: {cfg.OCR_LANG}")
        has_gpu = self._start()
        if has_gpu:
            print("[OK] GPU support detected — GPU acceleration enabled")
        elif has_gpu is False:
            print("[INFO] No GPU detected — running in CPU mode")
        print(f"[OK] OCR engine ready (worker pid {self._proc.pid}, "
              f"restart above {cfg.OCR_MAX_RSS_MB:.0f} MB)\n")

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start(self):
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child, self.cfg.OCR_LANG, self.cfg.OCR_DOC_PREPROCESS),
            daemon=True,
            name="paddleocr-worker",
        )
        proc.start()
        child.close()
        if not parent.poll(self.STARTUP_TIMEOUT):
            proc.kill()
            raise RuntimeError("[ERROR] OCR worker did not start in time")
        status, info = parent.recv()
        if status != "READY":
            proc.join(5)
            raise RuntimeError(f"[ERROR] OCR worker failed to load: {info}")
        self._proc, self._conn = proc, parent
        return info

    def _stop(self):
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if proc is None:
            return
        try:
            conn.send(None)
        except Exception:
            pass
        proc.join(5)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        try:
            conn.close()
        except Exception:
            pass

    def restart(self, reason: str = ""):
        rss = self.worker_rss_mb()
        t0 = time.time()
        self._stop()
        self._start()
        self.restarts += 1
        print(f"[OCR] worker restarted ({reason}; was {rss:.0f} MB) "
              f"in {time.time() - t0:.1f}s — restart #{self.restarts}")

    def worker_rss_mb(self) -> float:
        return _rss_mb(self._proc.pid) if self._proc is not None else 0.0

    def maintenance(self):
        """Call between batches: restart the worker if its RAM grew too much."""
        if self.worker_rss_mb() > self.cfg.OCR_MAX_RSS_MB:
            self.restart(f"RAM > {self.cfg.OCR_MAX_RSS_MB:.0f} MB")

    def close(self):
        self._stop()

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(self, image: Optional[np.ndarray]) -> List[str]:
        """Return the recognized text lines of a BGR image ([] on failure)."""
        if image is None:
            return []
        for attempt in (1, 2):
            try:
                if self._proc is None or not self._proc.is_alive():
                    self.restart("worker not running")
                self._conn.send(image)
                if not self._conn.poll(self.REQUEST_TIMEOUT):
                    raise TimeoutError("no answer from OCR worker")
                status, payload = self._conn.recv()
                if status == "OK":
                    return payload
                print(f"[WARN] OCR recognition error: {payload}")
                return []
            except Exception as e:
                if attempt == 2:
                    print(f"[WARN] OCR worker failure: {e}")
                    return []
                # Crashed / hung worker: start a fresh one and retry once
                try:
                    self.restart(f"failure: {e}")
                except Exception as e2:
                    print(f"[WARN] OCR worker restart failed: {e2}")
                    return []
        return []
