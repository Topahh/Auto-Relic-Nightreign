# engine/flow.py

# engine/flow.py

from __future__ import annotations

import time
from utils.log import dbg
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from engine.state_machine import Action, FlowContext


@dataclass
class FlowHooks:
    """
    Hooks implemented by ForgeBot and injected into ForgeFlow,
    so Flow stays modular and does not depend on the whole main class.
    """
    capture_text: Callable[[], Tuple[np.ndarray, list, str]]
    process_item: Callable[[int], Tuple[bool, str, np.ndarray]]
    ensure_relic_menu: Callable[[int, int], bool]


class ForgeFlow:
    """
    Execution layer:
    - executes actions decided by the state machine
    - updates flow context
    - delegates OCR/matching work to ForgeBot through hooks
    """

    def __init__(self, cfg, keyboard, hooks: FlowHooks):
        self.cfg = cfg
        self.keyboard = keyboard
        self.hooks = hooks

    def execute(self, action: Action, ctx: FlowContext) -> bool:
        """
        Execute one high-level action.
        Returns True when the action succeeded well enough to continue.
        Returns False when a blocking failure happened.
        """
        if action == Action.WAIT:
            time.sleep(self.cfg.POLL_INTERVAL)
            return True

        if action == Action.ENTER_FLATSTONE:
            dbg("[FLOW] MAIN_MENU -> press interact to enter flatstone")
            self.keyboard.press(self.cfg.KEY_INTERACT)
            time.sleep(self.cfg.WAIT_MENU)
            return True

        if action == Action.CHOOSE_BATCH_SIZE:
            if time.monotonic() - ctx.last_choose_time < self.cfg.F2_RETRY_DELAY:
                dbg("[FLOW] F2 sent recently -> wait for 10/10 instead of re-sending")
                time.sleep(self.cfg.POLL_INTERVAL)
                return True
            dbg(f"[FLOW] FLATSTONE_MENU -> choose batch size with {self.cfg.KEY_CHOSE_10_RELICS}")
            self.keyboard.press(self.cfg.KEY_CHOSE_10_RELICS)
            ctx.last_choose_time = time.monotonic()
            time.sleep(self.cfg.WAIT_MENU)
            return True

        if action == Action.CONFIRM_FLATSTONE:
            dbg("[FLOW] FLATSTONE_MENU -> confirm purchase/opening")
            self.keyboard.press(self.cfg.KEY_INTERACT)
            ctx.last_confirm_time = time.monotonic()
            # Short pause only: the next ticks watch the purchase screen go away
            # and send the skip as soon as the opening animation starts.
            time.sleep(self.cfg.WAIT_ANIM)
            return True

        if action == Action.RETRY_CONFIRM:
            if time.monotonic() - ctx.last_confirm_time < self.cfg.CONFIRM_RETRY_DELAY:
                time.sleep(self.cfg.POLL_INTERVAL)
                return True
            dbg("[FLOW] purchase screen frozen -> re-send confirm")
            self.keyboard.press(self.cfg.KEY_INTERACT)
            ctx.last_confirm_time = time.monotonic()
            return True

        if action == Action.SKIP_TO_RELIC_MENU:
            dbg("[FLOW] TRANSITION/FLATSTONE -> skip or advance with interact")
            self.keyboard.press(self.cfg.KEY_INTERACT)
            ctx.last_skip_time = time.monotonic()
            return True

        if action == Action.RETRY_SKIP:
            # Opening animation still on screen: poll it, and re-send the skip
            # only if it has not gone away after skip_retry_delay seconds.
            if time.monotonic() - ctx.last_skip_time < self.cfg.SKIP_RETRY_DELAY:
                time.sleep(self.cfg.POLL_INTERVAL)
                return True
            dbg("[FLOW] animation still running -> re-send skip")
            self.keyboard.press(self.cfg.KEY_INTERACT)
            ctx.last_skip_time = time.monotonic()
            return True

        if action == Action.PROCESS_CURRENT_RELIC:
            next_index = ctx.processed_relics + 1
            dbg(f"[FLOW] RELIC_MENU -> process relic {next_index}/{ctx.batch_size}")

            ok, current_text, _ = self.hooks.process_item(next_index)
            if not ok:
                print(f"[FLOW] Failed to validate relic {next_index}")
                return False

            ctx.processed_relics += 1
            dbg(f"[FLOW] Relic processed -> total={ctx.processed_relics}/{ctx.batch_size}")
            return True

        if action == Action.EXIT_RELIC_MENU:
            # While the menu is closing it still looks like the relic menu:
            # a re-sent F there lands on the main menu / purchase screen.
            if time.monotonic() - ctx.last_exit_time < self.cfg.EXIT_RETRY_DELAY:
                dbg("[FLOW] exit sent recently -> wait for the menu to close")
                time.sleep(self.cfg.POLL_INTERVAL)
                return True
            dbg("[FLOW] RELIC_MENU -> batch complete, exit menu")
            self.keyboard.press(self.cfg.KEY_INTERACT)
            ctx.last_exit_time = time.monotonic()
            return True

        if action == Action.RECOVER_TO_RELIC_MENU:
            dbg("[FLOW] Unknown state during active round -> ensure relic menu")
            ok = self.hooks.ensure_relic_menu(ctx.processed_relics + 1, max_retries=3)
            return ok

        if action == Action.RESET_ROUND:
            dbg("[FLOW] Reset round context")
            pending_new_round = ctx.pending_new_round
            main_menu_grace_ticks = ctx.main_menu_grace_ticks
            last_exit_time = ctx.last_exit_time
            ctx.reset_round()
            ctx.pending_new_round = pending_new_round
            ctx.main_menu_grace_ticks = main_menu_grace_ticks
            # keep it: a late "menu still closing" frame must not trigger a new F
            ctx.last_exit_time = last_exit_time
            return True

        print(f"[FLOW] Unhandled action: {action}")
        return False