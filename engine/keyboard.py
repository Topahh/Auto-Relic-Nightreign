# engine/keyboard.py
# Keyboard input controller — cross-platform.
#
# Backend selection happens ONCE at __init__ via method binding:
#   Linux Wayland / X11  →  xdotool  (original logic, untouched)
#   Windows              →  pydirectinput
#
# After __init__, self.press / self._focus_game / self.warmup_permissions
# point directly to the right implementation — no runtime branching.

import os
import time
import subprocess
import threading

from config.settings import Config
from engine.platform_detect import is_windows


# ------------------------------------------------------------------
# Keyboard controller
# ------------------------------------------------------------------


class GameWindowLost(Exception):
    """The game window disappeared (crash / closed): stop instead of typing
    keys into whatever window has the focus. Not a RuntimeError on purpose:
    the main loop retries RuntimeErrors forever."""


class KeyboardController:
    """
    Sends keystrokes to the game window.

    On Linux (Wayland / X11) : uses xdotool, targets the window by title.
    On Windows               : uses pydirectinput (DirectX-compatible).

    The correct implementation is bound to self.press / self._focus_game /
    self.warmup_permissions at __init__ time — no per-call branching.
    All timing values are read from cfg — no hardcoded sleeps.
    """

    _KEY_MAP_XDOTOOL = {
        'down': 'Down', 'up': 'Up', 'right': 'Right', 'left': 'Left',
        'enter': 'Return', 'return': 'Return', 'escape': 'Escape',
        'esc': 'Escape', 'space': 'space', 'tab': 'Tab',
        'f': 'f', 'f2': 'F2',
        '2': '2', '3': '3',
    }

    _KEY_MAP_PDI = {
        'down': 'down', 'up': 'up', 'right': 'right', 'left': 'left',
        'enter': 'enter', 'return': 'enter', 'escape': 'escape',
        'esc': 'escape', 'space': 'space', 'tab': 'tab',
        'f': 'f', 'f2': 'f2',
        '2': '2', '3': '3',
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Dispatch ONCE — no branching at call time
        if is_windows():
            self._init_windows()
        else:
            self._init_linux()


    # ------------------------------------------------------------------
    # Backend initializers
    # ------------------------------------------------------------------

    def _init_linux(self):
        """Set up the xdotool backend and bind methods to Linux implementations."""
        self._env = {**os.environ, 'DISPLAY': ':0'}
        self._window_id = None
        self._verify_and_find_window()
        # Bind methods
        self.press              = self._press_linux
        self._focus_game        = self._focus_game_linux
        self.warmup_permissions = self._warmup_linux
        self.release_all        = self._release_all_linux

    def _init_windows(self):
        """Set up the pydirectinput backend and bind methods to Windows implementations."""
        try:
            import pydirectinput as _pdi
            _pdi.PAUSE = 0.02
            self._pdi = _pdi
        except ImportError:
            raise RuntimeError(
                "[ERROR] pydirectinput is not installed. "
                "Install it with: pip install pydirectinput"
            )
        print("[OK] pydirectinput backend ready")
        # Bind methods
        self.press              = self._press_windows
        self._focus_game        = self._focus_game_windows
        self.warmup_permissions = self._warmup_windows
        self.release_all        = self._release_all_windows


    # ------------------------------------------------------------------
    # Linux — initialization
    # ------------------------------------------------------------------

    def _verify_and_find_window(self):
        """
        Check that xdotool is installed, then locate the game window by title.

        Stores the window ID in self._window_id for use by _focus_game().
        Prints a warning if the window cannot be found — inputs may still work
        if the game window happens to have focus.
        """
        r = subprocess.run(['which', 'xdotool'], capture_output=True)
        if r.returncode != 0:
            print("[ERROR] xdotool not found — install it with: sudo dnf install xdotool")
            return
        print("[OK] xdotool detected")

        r = subprocess.run(
            ['xdotool', 'search', '--name', 'ELDEN RING NIGHTREIGN'],
            capture_output=True, text=True, env=self._env
        )

        if r.returncode == 0 and r.stdout.strip():
            self._window_id = r.stdout.strip().split('\n')[0]
            print(f"[OK] Game window found — ID {self._window_id}")
        else:
            print("[WARN] Game window not found — inputs may fail")


    # ------------------------------------------------------------------
    # Linux — focus & warmup
    # ------------------------------------------------------------------

    def _focus_game_linux(self):
        """Focus the game window before sending any input.
        Delay uses cfg.FOCUS_DELAY (from [Timing] focus_delay)."""
        if self._window_id:
            self._xdo('windowfocus', '--sync', self._window_id)
            time.sleep(self.cfg.FOCUS_DELAY)

    def _warmup_linux(self):
        """
        Trigger an early focus + harmless keypress to initialize input permissions.
        Delay steps use cfg.WARMUP_DELAY (from [Timing] warmup_delay).
        """
        self._focus_game()
        time.sleep(self.cfg.WARMUP_DELAY)
        self._xdo('key', 'Shift_L')
        time.sleep(self.cfg.WARMUP_DELAY)


    # ------------------------------------------------------------------
    # Linux — key press primitive
    # ------------------------------------------------------------------

    def _press_linux(self, key: str, delay: float = None, hold: float = None):
        """
        Send a controlled keypress via xdotool.
        - hold  : key down duration  (if None, uses self.cfg.KEY_HOLD)
        - delay : wait after release (if None, uses self.cfg.KEY_INTERVAL)
        """
        xkey   = self._KEY_MAP_XDOTOOL.get(key.lower(), key)
        after  = delay if delay is not None else self.cfg.KEY_INTERVAL
        hold_t = hold  if hold  is not None else self.cfg.KEY_HOLD

        # One xdotool process for focus + down + hold + up (was 3 processes
        # + a Python sleep). No --clearmodifiers: it re-pressed the modifiers
        # you physically held (Ctrl during Ctrl+C), leaving Ctrl "stuck".
        focus = (['windowfocus', '--sync', self._window_id, 'sleep', str(self.cfg.FOCUS_DELAY)]
                 if self._window_id else [])
        try:
            r = self._xdo(*focus, 'keydown', xkey, 'sleep', str(hold_t), 'keyup', xkey)
            if r.returncode != 0 and self._window_id and not self._window_exists():
                # xdotool aborts the chain at the failed windowfocus: no key was sent
                raise GameWindowLost(f"game window {self._window_id} no longer exists")
        except GameWindowLost:
            raise
        except BaseException:
            # Interrupted mid-press (Ctrl+C): make sure the key goes back up
            self._xdo('keyup', xkey)
            raise
        time.sleep(after)

    def _xdo(self, *args):
        """Run xdotool in its own session so a terminal Ctrl+C cannot kill it
        halfway through (which would leave a key pressed)."""
        return subprocess.run(
            ['xdotool', *args], env=self._env, capture_output=True,
            start_new_session=True,
        )

    def _window_exists(self) -> bool:
        return self._xdo('getwindowname', self._window_id).returncode == 0

    _RELEASE_ON_EXIT_LINUX = (
        'Control_L', 'Control_R', 'Shift_L', 'Shift_R', 'Alt_L', 'Alt_R',
        'Super_L', 'Super_R', 'ISO_Level3_Shift',
    )

    def _release_all_linux(self):
        keys = set(self._RELEASE_ON_EXIT_LINUX)
        for k in (self.cfg.KEY_INTERACT, self.cfg.KEY_DOWN, self.cfg.KEY_RIGHT,
                  self.cfg.KEY_KEEP, self.cfg.KEY_DISCARD, self.cfg.KEY_CHOSE_10_RELICS):
            if k:
                keys.add(self._KEY_MAP_XDOTOOL.get(k.lower(), k))
        self._xdo('keyup', *sorted(keys))


    # ------------------------------------------------------------------
    # Windows — focus & warmup  (no-ops)
    # ------------------------------------------------------------------

    def _focus_game_windows(self):
        pass  # pydirectinput sends input at OS level, no window focus needed

    def _warmup_windows(self):
        time.sleep(self.cfg.WARMUP_DELAY)


    # ------------------------------------------------------------------
    # Windows — key press primitive
    # ------------------------------------------------------------------

    def _press_windows(self, key: str, delay: float = None, hold: float = None):
        """
        Send a controlled keypress via pydirectinput.
        - hold  : key down duration  (if None, uses self.cfg.KEY_HOLD)
        - delay : wait after release (if None, uses self.cfg.KEY_INTERVAL)
        """
        mapped = self._KEY_MAP_PDI.get(key.lower(), key)
        after  = delay if delay is not None else self.cfg.KEY_INTERVAL
        hold_t = hold  if hold  is not None else self.cfg.KEY_HOLD

        self._pdi.keyDown(mapped)
        try:
            time.sleep(hold_t)
        finally:
            self._pdi.keyUp(mapped)
        time.sleep(after)

    def _release_all_windows(self):
        keys = {'ctrl', 'shift', 'alt', 'win'}
        for k in (self.cfg.KEY_INTERACT, self.cfg.KEY_DOWN, self.cfg.KEY_RIGHT,
                  self.cfg.KEY_KEEP, self.cfg.KEY_DISCARD, self.cfg.KEY_CHOSE_10_RELICS):
            if k:
                keys.add(self._KEY_MAP_PDI.get(k.lower(), k))
        for k in keys:
            try:
                self._pdi.keyUp(k)
            except Exception:
                pass


    # ------------------------------------------------------------------
    # Public stubs — replaced by method binding at __init__ time.
    # Defined here only so IDEs / type checkers see the signatures.
    # ------------------------------------------------------------------

    def press(self, key: str, delay: float = None, hold: float = None):
        raise NotImplementedError("Backend not initialized")

    def _focus_game(self):
        raise NotImplementedError("Backend not initialized")

    def warmup_permissions(self):
        raise NotImplementedError("Backend not initialized")

    def release_all(self):
        """Release modifiers and every bot key (called on shutdown)."""
        raise NotImplementedError("Backend not initialized")


    #region Game actions
    # ------------------------------------------------------------------
    # Game actions  (identical on both backends — delegate to self.press)
    # ------------------------------------------------------------------

    def keep_item(self):
        """Press the keep key."""
        self.press(self.cfg.KEY_KEEP)

    def discard_item(self):
        """Press the discard key."""
        self.press(self.cfg.KEY_DISCARD)

    def press_interact(self):
        self.press(self.cfg.KEY_INTERACT)

    def press_choose_10(self):
        self.press(self.cfg.KEY_CHOSE_10_RELICS)

    def press_keep(self):
        self.press(self.cfg.KEY_KEEP)

    def press_discard(self):
        self.press(self.cfg.KEY_DISCARD)

    def press_interact_and_wait_short(self):
        self.press(self.cfg.KEY_INTERACT)
        time.sleep(self.cfg.WAIT_ANIM)

    def press_interact_and_wait_long(self):
        self.press(self.cfg.KEY_INTERACT)
        time.sleep(self.cfg.WAIT_ANIM_EXTRA)
    # endregion


# ------------------------------------------------------------------
# Helper I/O (shared utility)
# ------------------------------------------------------------------


def _readline_with_timeout(pipe, timeout=45):
    """
    Read one line from a pipe with a hard timeout.

    Runs the blocking readline() in a daemon thread.
    Raises TimeoutError if no response arrives within `timeout` seconds.
    """
    result = {"line": None, "error": None}

    def _target():
        try:
            result["line"] = pipe.readline()
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError("Timed out waiting for helper response")

    if result["error"] is not None:
        raise result["error"]

    return result["line"]