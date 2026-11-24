# main.py
import os
import signal
import time
import datetime
import cv2
import numpy as np
from typing import Tuple, Optional

from config.settings import Config
from engine.ocr import OCREngine
from engine.keyboard import KeyboardController, GameWindowLost
from engine.capture import ScreenCapture
from engine.matcher import ItemMatcher
from engine.flow import ForgeFlow, FlowHooks
from engine.state_machine import ForgeBotStateMachine, FlowContext, TickInput, UIState, Action
# from engine.locker import CurrencyLocker
from utils.text import fuzzy_clean_text, natural_sort_key
from utils.stats import Statistics
from utils.signal import StopSignal
from utils.log import dbg


# region ForgeBot
class ForgeBot:
    """Forge Bot — OCR synchronized, fully INI-configured."""

    def __init__(self):
        self.cfg = Config()
        self.ocr = OCREngine(self.cfg)
        self.keyboard = KeyboardController(self.cfg)
        # Last OCR'd screen, reused until a key is pressed (or the next tick).
        self._frame_cache = None
        self._prefetched = False
        self._cache_settled = False   # cache passed the stability check
        _raw_press = self.keyboard.press

        def _press_and_invalidate(*args, **kwargs):
            self._frame_cache = None
            self._prefetched = False
            return _raw_press(*args, **kwargs)

        self.keyboard.press = _press_and_invalidate
        self.matcher = ItemMatcher(self.cfg)
        self.capture = ScreenCapture(self.cfg.SCAN_REGION)
        if self.cfg.DEBUG:
            self.debug_screenshot()
        self.stats = Statistics(
            self.cfg.lang,
            max_in_memory_details=300,
            flush_every=300,
            temp_path="/tmp/hajiwo_stats.jsonl",
        )
        self.log_path = None
        self.audit_path = None   # one line per relic: OCR text + decision
        self._audit_bytes = 0
        self.last_stats_flush_scanned = 0
        self.stop_signal = StopSignal(self.cfg.lang)
        self.stop_signal.clear()
        self.locker = None
        self.first_round = True
        self.flow_context = FlowContext(batch_size=self.cfg.BATCH_SIZE)
        self.state_machine = ForgeBotStateMachine(
            relic_tokens=self.cfg.RELIC_TOKENS,
            flatstone_tokens=self.cfg.FLATSTONE_TOKENS,
            main_menu_tokens=self.cfg.MAIN_MENU_TOKENS,
            reset_menu_tokens=self.cfg.RESET_MENU_TOKENS,
            min_text_len=self.cfg.MIN_TEXT_LEN,  # Seuil global state machine
            batch_size=self.cfg.BATCH_SIZE,
        )
        self.flow = ForgeFlow(
            cfg=self.cfg,
            keyboard=self.keyboard,
            hooks=FlowHooks(
                capture_text=self.capture_text,
                process_item=self.process_item,
                ensure_relic_menu=self.ensure_relic_menu,
            ),
        )

    def flush_stats_if_needed(self, force: bool = False):
        """
        Periodically append stats to disk and free detailed OCR lists.
        Counters remain intact.
        """
        if not self.log_path:
            return

        scanned_delta = self.stats.scanned - self.last_stats_flush_scanned
        should_flush = force or scanned_delta >= 150

        if not should_flush:
            return

        suffix = f"snapshot #{self.stats.flush_count + 1} @ scanned={self.stats.scanned}"
        self.stats.append_log_snapshot(
            self.log_path,
            header_suffix=suffix,
            include_details=True,
            clear_after_write=True,
        )
        self.last_stats_flush_scanned = self.stats.scanned

# endregion
# region 
    # ==================== SYNCHRONIZATION ====================

    def ensure_relic_menu(self, index: int, max_retries: int = 5) -> bool:
        for attempt in range(1, max_retries + 1):
            _, texts, recognized = self.capture_text()
            dbg(f" [{index:2d}] [ENSURE] attempt {attempt}/{max_retries} -> '{recognized}'")

            if self.is_relic_menu(recognized):
                return True

            if self.is_flatstone_menu(recognized):
                dbg(f" [{index:2d}] [ENSURE] flatstone -> enter relic menu")
                self.keyboard.press(self.cfg.KEY_INTERACT)
                time.sleep(self.cfg.WAIT_ANIM_EXTRA)
                continue

            if self.is_main_menu(recognized):
                dbg(f" [{index:2d}] [ENSURE] main menu -> open flatstone path")
                self.keyboard.press(self.cfg.KEY_INTERACT)
                time.sleep(self.cfg.WAIT_ANIM_EXTRA)
                continue

            if self.is_partial_relic_overlay(recognized):
                dbg(f" [{index:2d}] [ENSURE] partial overlay -> press F to dismiss")
                self.keyboard.press(self.cfg.KEY_INTERACT)
                time.sleep(max(self.cfg.RECOVER_DELAY, self.cfg.POLL_INTERVAL))
                continue

            dbg(f" [{index:2d}] [ENSURE] unknown/empty UI -> press F to recover")
            self.keyboard.press(self.cfg.KEY_INTERACT)
            time.sleep(max(self.cfg.RECOVER_DELAY, self.cfg.POLL_INTERVAL))

        return False
    
    def is_main_menu(self, text: str) -> bool:
        tick = self.state_machine.detect_state(text, self.flow_context)
        state = self.state_machine.classify_state(tick, self.flow_context)
        return state == UIState.MAIN_MENU

    def is_flatstone_menu(self, text: str) -> bool:
        tick = self.state_machine.detect_state(text, self.flow_context)
        state = self.state_machine.classify_state(tick, self.flow_context)
        return state == UIState.FLATSTONE_MENU
    
    def is_relic_menu(self, text: str) -> bool:
        tick = self.state_machine.detect_state(text, self.flow_context)
        state = self.state_machine.classify_state(tick, self.flow_context)
        return state == UIState.RELIC_MENU
    
    def is_reset_menu(self, text: str) -> bool:
        tick = self.state_machine.detect_state(text, self.flow_context)
        state = self.state_machine.classify_state(tick, self.flow_context)
        return state == UIState.RESET_MENU

    def is_partial_relic_overlay(self, text: str) -> bool:
        t = text.lower()
        return (
            "close" in t
            and "reset" in t
            and "addremovefavorites" not in t
            and "sellnow" not in t
        )

    def looks_like_action_menu(self, text: str) -> bool:
        t = (text or "").lower()
        
        # NOUVEAU : ignore si fin de batch
        if self.flow_context.processed_relics >= self.cfg.BATCH_SIZE - 2:
            return False
        
        # NOUVEAU : ignore si contient "reset" ET "favorites/sell"
        if "reset" in t and ("favorites" in t or "sellnow" in t):
            return False
        
        # Tes règles existantes...
        has_overlay_tokens = ("close" in t and "reset" in t)
        has_relic_tokens = any(tok in t for tok in self.cfg.RELIC_TOKENS)
        return has_overlay_tokens and not has_relic_tokens and len(t) < 40

    def capture_text(self, use_cache: bool = True) -> Tuple[np.ndarray, list, str]:
        """Capture + OCR. Reuses the last result while no key was pressed:
        tick(), ensure_relic_menu() and process_item() used to OCR the very
        same screen three times in a row."""
        if use_cache and self._frame_cache is not None:
            return self._frame_cache
        image = self.capture.capture()
        texts = self.ocr.recognize(image) if image is not None else []
        recognized = "".join(fuzzy_clean_text(t) for t in texts)
        self._frame_cache = (image, texts, recognized)
        self._cache_settled = False
        return self._frame_cache

    def _prefetch(self, frame, texts, text):
        """The screen found by wait_for_next_item becomes the next tick's
        screen: no need to capture + OCR it again.
        It is first checked for stability, so a relic caught mid-redraw is
        never matched on half its text (re-OCR only if the image moved)."""
        for _ in range(3):
            if frame is None:
                break
            time.sleep(self.cfg.SETTLE_DELAY)
            check = self.capture.capture()
            if check is None or self._same_frame(check, frame):
                break
            check_texts = self.ocr.recognize(check)
            check_text = "".join(fuzzy_clean_text(t) for t in check_texts)
            frame, texts = check, check_texts
            if check_text == text:
                break
            dbg(f"  [SYNC] screen still changing: '{text[:50]}' -> '{check_text[:50]}'")
            text = check_text
        self._frame_cache = (frame, texts, text)
        self._prefetched = True
        self._cache_settled = True
        return frame, text

    @staticmethod
    def _same_frame(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
        """Pixel-identical (or nearly) frames -> the OCR result can be reused."""
        if a is None or b is None or a.shape != b.shape:
            return False
        return float(np.mean(cv2.absdiff(a, b))) < 0.5

    def frame_has_changed(
        self,
        prev_frame: np.ndarray,
        new_frame: np.ndarray,
        threshold: float = None
    ) -> bool:
        """Visual validation: has the screen changed?"""
        if prev_frame is None or new_frame is None:
            return True
        threshold = threshold if threshold is not None else self.cfg.SYNC_THRESHOLD
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray, new_gray = prev_frame, new_frame
        diff = cv2.absdiff(prev_gray, new_gray)
        mean_diff = np.mean(diff)
        changed = mean_diff > threshold
        dbg(f"  [SYNC] frame_diff={mean_diff:.1f}>{threshold}={changed}")
        return changed

    def is_valid_ocr_text(
        self,
        text: str,
        prev_text: str = None,
        min_len: int = None,
        allow_unchanged: bool = False
    ) -> bool:
        """Textual validation: does the OCR text look valid enough to consider a real menu change?"""
        min_len = min_len if min_len is not None else self.cfg.MIN_TEXT_LEN
        if not text or len(text) < min_len:
            return False
        if not allow_unchanged and prev_text and text == prev_text:
            return False
        if len(set(text)) / len(text) < 0.3:
            return False
        return True
    
    def wait_for_next_item(
        self,
        prev_frame: np.ndarray,
        prev_text: str,
        allow_unchanged_text: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Wait for the next valid item frame."""
        timeout = self.cfg.SYNC_TIMEOUT
        poll_interval = self.cfg.POLL_INTERVAL
        empty_reads_max = self.cfg.EMPTY_READS_MAX
        start_time = time.time()
        empty_reads = 0

        dbg(f"  [SYNC] Waiting next valid item (timeout={timeout}s)...")
        last_poll_frame, last_poll_texts = None, None
        while time.time() - start_time < timeout:
            if self.stop_signal.should_stop():
                return None, None

            new_frame = self.capture.capture()
            if new_frame is None:
                new_texts = []
            elif self._same_frame(new_frame, last_poll_frame):
                new_texts = last_poll_texts      # screen did not move: no new OCR
            else:
                new_texts = self.ocr.recognize(new_frame)
            last_poll_frame, last_poll_texts = new_frame, new_texts
            new_text = "".join([fuzzy_clean_text(t) for t in new_texts])
            if not self.frame_has_changed(prev_frame, new_frame):
                if len(new_text) >= self.cfg.RELIC_MIN_LEN:
                    if allow_unchanged_text:
                        return self._prefetch(new_frame, new_texts, new_text)
                    elif new_text != prev_text:
                        return self._prefetch(new_frame, new_texts, new_text)
                time.sleep(poll_interval)
                continue

            dbg("  [SYNC] frame changed → OCR...")

            if self.looks_like_action_menu(new_text):
                dbg(f"  [SYNC] Overlay/action popup detected: '{new_text}' -> skip")
                time.sleep(poll_interval)
                continue

            if (
                len(new_text) >= self.cfg.RELIC_MIN_LEN
                and new_text != prev_text    # Changement de texte
                and not self.looks_like_action_menu(new_text)  # Pas overlay
            ):
                dbg(f"  [SYNC] ✓ FALLBACK ✓ len={len(new_text)} != prev: '{new_text[:60]}...'")
                return self._prefetch(new_frame, new_texts, new_text)

            if self.is_relic_menu(new_text) and self.is_valid_ocr_text(new_text, prev_text):
                dbg(f"  [SYNC] ✓ Next relic menu detected: '{new_text}'")
                return self._prefetch(new_frame, new_texts, new_text)

            if self.is_main_menu(new_text):
                dbg(f"  [SYNC] main menu detected during sync: '{new_text}'")
                return self._prefetch(new_frame, new_texts, new_text)

            if not self.is_valid_ocr_text(
                new_text,
                prev_text,
                self.cfg.RELIC_MIN_LEN,
                allow_unchanged=allow_unchanged_text
            ):
                dbg(f"  [SYNC] Invalid OCR: '{new_text}' (len={len(new_text)})")
                empty_reads += 1
                if empty_reads >= empty_reads_max:
                    dbg(f"  [SYNC] {empty_reads} empty reads → end of list")
                    return None, None
                continue

            dbg(f"  [SYNC] ✓ Valid item: '{new_text}'")
            return self._prefetch(new_frame, new_texts, new_text)

        dbg("  [SYNC] timeout → end of list")
        return None, None

# endregion
# region processing items
    # ==================== PROCESS_ITEM ====================

    def process_item(self, index: int) -> Tuple[bool, str, np.ndarray]:
        """
        Ensure relic menu, OCR current relic, decide keep/discard,
        then wait for a real transition to the next relic before returning success.

        Returns:
            (success, recognized_text, image)
        """
        if not self.ensure_relic_menu(index):
            print(f" [{index:2d}] [ERROR] UI not recovered -> skip item")
            return False, "", None

        image, texts, recognized = self.capture_text()
        if not self._cache_settled:
            # e.g. 1st relic right after the opening animation: make sure the
            # text is fully drawn before matching it
            image, recognized = self._prefetch(image, texts, recognized)
            texts = self._frame_cache[1]

        dbg(f" [{index:2d}] OCR: '{recognized}' (len={len(recognized)})")

        keep, info, matched_kw, blacklist_kw, has_partial, group_name, all_results = \
            self.matcher.match_detailed(texts)
        self.stats.scanned += 1

        if keep:
            outcome = "kept"
        elif has_partial and blacklist_kw:
            outcome = "blacklisted"
        elif has_partial:
            outcome = "partial"
        else:
            outcome = "discard"

        self.write_audit(index, outcome, group_name, recognized, blacklist_kw)

        if self.cfg.RICH_STATS:
            self.stats.record_relic(
                batch=self.stats.rounds,
                index=index,
                outcome=outcome,
                group=group_name or "",
                all_results=all_results,
                matched_kw=matched_kw or [],
            )

        if keep:
            self.stats.kept += 1
            self.stats.add_kept_item(texts, matched_kw, group_name)
            print(f" [{index:2d}] ★ KEEP  {info}")
            self.keyboard.keep_item()
        else:
            if blacklist_kw:
                print(f" [{index:2d}] ✗ blacklist: {', '.join(blacklist_kw)}")
            else:
                print(f" [{index:2d}] ✗")
            dbg(f"      {info}")

            if has_partial and blacklist_kw:
                self.stats.add_qualified_blacklisted(texts, matched_kw, blacklist_kw)
            elif has_partial and not blacklist_kw:
                self.stats.add_partial_match(texts, matched_kw, info, group_name)

            self.keyboard.discard_item()

        # For the last relic of the batch, do not wait for a "next item"
        if index >= self.cfg.BATCH_SIZE:
            dbg(f" [{index:2d}] [SYNC] last relic in batch -> no next-item wait needed")
            return True, recognized, image

        dbg(f" [{index:2d}] [SYNC] waiting for next relic...")
        next_frame, next_text = self.wait_for_next_item(image, recognized, allow_unchanged_text=False)

        if next_frame is None or not next_text:
            dbg(f" [{index:2d}] [SYNC] next relic not observed -> action not validated")
            self.flow_context.max_failed_syncs += 1
            # if self.flow_context.max_failed_syncs >= 3 and index >= 8:
            if index >= self.cfg.BATCH_SIZE - 2:
                print(f"[END] {self.flow_context.max_failed_syncs} failed syncs → end of batch")
                self.flow_context.round_active = False
                self.flow_context.pending_new_round = True
                return True, recognized, image
            return False, recognized, image

        dbg(f" [{index:2d}] [SYNC] next relic detected: '{next_text}'")
        return True, recognized, image

# region flow and state machine
    # ==================== GAME ACTIONS ====================

    def run_round(self) -> bool:
        raise NotImplementedError("run_round is deprecated; use tick() and the state machine flow")

    # ==================== RUN ====================

    def start_currency_locker(self) -> bool:
        print("[STEP 1] Linux mode: Skipping currency lock\n")
        return True

    def show_config_keywords(self):
        if not self.cfg.DEBUG:
            groups = sorted(self.cfg.KEYWORD_GROUPS.keys(), key=natural_sort_key)
            print(f"Groups ({len(groups)}): {', '.join(groups) or 'none'}")
            return
        print("\n" + "-" * 50)
        print("Keyword Groups")
        print("-" * 50)
        if not self.cfg.KEYWORD_GROUPS:
            print("No keyword groups")
        else:
            for group_name in sorted(self.cfg.KEYWORD_GROUPS.keys(), key=natural_sort_key):
                group_config = self.cfg.KEYWORD_GROUPS[group_name]
                print(f"\n【{group_name}】")
                if group_config["a"]:
                    print(f"  Required (≥{group_config['min']}): {chr(32).join(group_config['a'])}")
                if group_config["b"]:
                    print(f"  Optional : {chr(32).join(group_config['b'])}")
                if group_config["blacklist"]:
                    print(f"  Blacklist: {chr(32).join(group_config['blacklist'])}")
        print("=" * 50)

    def wait_user_ready(self) -> bool:
        print("\n[STEP 2] Prepare...")
        print("=" * 50)
        print("Steps:")
        print("  1. Enter shop")
        print("  2. Select relic batch (10)")
        print("  3. Press Enter")
        print("\nPress ESC to stop anytime")
        print("=" * 50)
        print("\nPress Enter to continue...")
        input()
        return True

    def tick(self) -> bool:
        # Fresh screen each tick, unless wait_for_next_item just read it
        if not self._prefetched:
            self._frame_cache = None
        self._prefetched = False
        _, _, recognized = self.capture_text()
        dbg(
            f"[OCR] len={len(recognized)} grace={self.flow_context.main_menu_grace_ticks} "
            f"transition={self.flow_context.transition_ticks} pending={self.flow_context.pending_new_round} "
            f"text='{recognized[:120]}'"
        )
        decision = self.state_machine.decide(recognized, self.flow_context)
        dbg(f"[STATE] state={decision.state} action={decision.action} reason={decision.reason}")
        self.state_machine.apply_updates(self.flow_context, decision)
        if decision.action == Action.RESET_ROUND and self.flow_context.processed_relics > 0:
            self.stats.rounds += 1
            print(self.stats.batch_line())
            # Between two batches: restart the OCR worker if PaddleOCR's leak
            # pushed it past [OCR] max_rss_mb (weeks-long sessions)
            self.ocr.maintenance()
        return self.flow.execute(decision.action, self.flow_context)

    def run(self):
        """Main loop."""
        print("=" * 50)
        print("Relic Auto-Forging — OCR SYNC")
        print("=" * 50)

        if not self.cfg.KEYWORD_GROUPS:
            print("[ERROR] No keywords configured")
            self.wait_for_exit()
            return

        if not self.start_currency_locker():
            print("[ERROR] Currency lock failed")
            self.wait_for_exit()
            return

        print("[INIT] Warmup input permissions...")
        self.keyboard.warmup_permissions()
        print("[INIT] Warmup done")

        self.show_config_keywords()
        if not self.wait_user_ready():
            self.wait_for_exit()
            return

        print("\nSwitch to game...")
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        # Archivage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"hajiwo_log_{timestamp}.txt"
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_captures")
        os.makedirs(script_dir, exist_ok=True)
        self.log_path = os.path.join(script_dir, log_filename)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("[SESSION START]\n\n")
        self.audit_path = os.path.join(script_dir, f"hajiwo_relics_{timestamp}.tsv")
        self._audit_bytes = 0   # header written with the first relic

        try:
            while not self.stop_signal.should_stop():
                try:
                    if not self.tick():
                        print("[WARNING] Tick failed, retrying...")
                        time.sleep(1)

                    self.flush_stats_if_needed(force=False)

                except GameWindowLost as e:
                    print(f"\n[STOP] {e} — the game is closed or crashed, stopping the bot")
                    break

                except RuntimeError as e:
                    print(f"[WARNING] Transient runtime error in tick: {e} — retrying in 1s")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

        finally:
            # A second Ctrl+C must not abort the cleanup half-way
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.stop_signal.event.set()
            self.release_inputs()
            time.sleep(0.2)

            if self.locker:
                self.locker.stop()

            try:
                self.flush_stats_if_needed(force=True)
            except Exception as e:
                print(f"[WARNING] Failed to flush stats on shutdown: {e}")

            # Flush résiduel du buffer de records structurés
            try:
                self.stats.flush_final()
            except Exception as e:
                print(f"[WARNING] Failed to flush record buffer: {e}")

            self.stats.print_report()
            if self.cfg.RICH_STATS:
                self.stats.print_rich_summary()

            if self.log_path:
                try:
                    self.stats.append_log_snapshot(
                        self.log_path,
                        header_suffix="final summary",
                        include_details=True,
                        clear_after_write=False,
                        final=True,
                    )
                    if self.cfg.RICH_STATS:
                        self.stats.save_rich_summary(self.log_path)
                    print(f"[OK] {self.cfg.lang.get('log_saved')}: {self.log_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to append final summary: {e}")

            # Nettoyage du fichier temp .jsonl
            try:
                self.stats.cleanup_temp()
            except Exception as e:
                print(f"[WARNING] Failed to cleanup temp stats file: {e}")

            self.shutdown()

    def write_audit(self, index, outcome, group_name, recognized, blacklist_kw):
        """Append one TSV line per relic (re-checkable with audit_relics.py)."""
        if not self.audit_path:
            return
        line = (f"{self.stats.rounds + 1}\t{index}\t{outcome}\t{group_name or ''}\t"
                f"{','.join(blacklist_kw or [])}\t{recognized}\n")
        try:
            # Rotation: weeks-long sessions must not fill the disk.
            # Past audit_max_mb the file becomes <name>.1.tsv (previous .1 dropped).
            if self._audit_bytes > self.cfg.AUDIT_MAX_MB * 1024 * 1024:
                os.replace(self.audit_path, self.audit_path[:-4] + ".1.tsv")
                self._audit_bytes = 0
            if self._audit_bytes == 0:
                header = "batch\tindex\toutcome\tgroup\tblacklist\tocr_text\n"
                if not os.path.exists(self.audit_path) or os.path.getsize(self.audit_path) == 0:
                    line = header + line
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(line)
            self._audit_bytes += len(line.encode("utf-8"))
        except Exception as e:
            dbg(f"[AUDIT] write failed: {e}")

    def release_inputs(self):
        """Release every key/modifier the bot may hold."""
        try:
            self.keyboard.release_all()
        except Exception as e:
            print(f"[WARNING] Failed to release keys: {e}")

    def shutdown(self):
        """Stop the capture helper and release keys. Safe to call twice."""
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True
        self.release_inputs()
        try:
            self.capture.close()
        except Exception as e:
            print(f"[WARNING] Failed to stop screen capture: {e}")
        try:
            self.ocr.close()
        except Exception as e:
            print(f"[WARNING] Failed to stop OCR worker: {e}")

    def debug_screenshot(self):
        img = self.capture.capture()
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "debug_captures",
            f"debug_{datetime.datetime.now().strftime('%H%M%S')}.png",
        )
        cv2.imwrite(path, img)
        print(f"[DEBUG] Saved: {path}")
        h, w = img.shape[:2]
        print(f"[DEBUG] captured size = {w}x{h}")
        texts = self.ocr.recognize(img)
        print(f"[DEBUG] OCR: {texts}")

    def wait_for_exit(self):
        input("\nPress Enter to exit...")

# endregion
# region main

def _sigterm_to_interrupt(signum, frame):
    # kill / closing the terminal -> same clean shutdown path as Ctrl+C
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
    signal.signal(signal.SIGHUP, _sigterm_to_interrupt)
    bot = None
    try:
        bot = ForgeBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if bot is not None:
            bot.shutdown()
        print("\n" + "=" * 50)
        if bot and bot.cfg and bot.cfg.lang:
            print(bot.cfg.lang.get("program_done"))
        else:
            print("Program completed")
        print("=" * 50)
        try:
            input("Press Enter to exit...")
        except (EOFError, KeyboardInterrupt):
            pass


# endregion