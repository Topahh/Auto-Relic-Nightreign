# engine/state_machine.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

# region Context and decision data structures
class UIState(str, Enum):
    MAIN_MENU = "MAIN_MENU"
    FLATSTONE_MENU = "FLATSTONE_MENU"
    RELIC_MENU = "RELIC_MENU"
    RESET_MENU = "RESET_MENU"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class Action(str, Enum):
    WAIT = "WAIT"
    ENTER_FLATSTONE = "ENTER_FLATSTONE"
    CHOOSE_BATCH_SIZE = "CHOOSE_BATCH_SIZE"
    CONFIRM_FLATSTONE = "CONFIRM_FLATSTONE"
    SKIP_TO_RELIC_MENU = "SKIP_TO_RELIC_MENU"
    PROCESS_CURRENT_RELIC = "PROCESS_CURRENT_RELIC"
    EXIT_RELIC_MENU = "EXIT_RELIC_MENU"
    RECOVER_TO_RELIC_MENU = "RECOVER_TO_RELIC_MENU"
    RESET_ROUND = "RESET_ROUND"
    RETRY_CONFIRM = "RETRY_CONFIRM"
    RETRY_SKIP = "RETRY_SKIP"


@dataclass
class TickInput:
    recognized_text: str
    min_text_len: int
    relic_token_hits: int = 0
    flatstone_token_hits: int = 0
    main_menu_token_hits: int = 0
    reset_token_hits: int = 0


@dataclass
class FlowContext:
    processed_relics: int = 0
    batch_size: int = 10
    max_failed_syncs: int = 0
    batch_select_requested: bool = False
    choose_10_done: bool = False
    confirm_done: bool = False
    batch_exit_sent: bool = False
    transition_started: bool = False
    skip_sent: bool = False
    recovering: bool = False
    round_active: bool = False
    pending_new_round: bool = False
    last_action: Optional[Action] = None
    last_state: Optional[UIState] = None
    empty_ticks: int = 0
    transition_ticks: int = 0
    main_menu_grace_ticks: int = 0
    # Ticks spent in a non-progressing WAIT (flatstone 10/10 wait, main menu
    # during an active round, unknown before purchase). Bounds those WAITs.
    stall_ticks: int = 0
    # Wall-clock time of the last exit / F2 press (set by ForgeFlow): retries
    # are spaced in SECONDS, not ticks — faster ticks must not mean more presses.
    last_exit_time: float = 0.0
    last_choose_time: float = 0.0
    last_confirm_time: float = 0.0
    last_skip_time: float = 0.0

    def reset_round(self) -> None:
        self.processed_relics = 0
        self.max_failed_syncs = 0
        self.batch_select_requested = False
        self.choose_10_done = False
        self.confirm_done = False
        self.batch_exit_sent = False
        self.transition_started = False
        self.skip_sent = False
        self.recovering = False
        self.round_active = False
        self.pending_new_round = False
        self.last_action = None
        self.empty_ticks = 0
        self.transition_ticks = 0
        self.stall_ticks = 0
        self.last_exit_time = 0.0
        self.last_choose_time = 0.0
        self.last_confirm_time = 0.0
        self.last_skip_time = 0.0

@dataclass
class TickDecision:
    state: UIState
    action: Action
    reason: str
    updates: Dict[str, Any] = field(default_factory=dict)

# endregion
# region definition and helper methods of the state machine

# Stall limits (in ticks) for WAITs that would otherwise never end
F2_RETRY_TICKS = 8          # 10/10 not seen after F2 -> send F2 again
MAIN_MENU_STALL_TICKS = 10  # real main menu seen during an active round -> restart round
UNKNOWN_STALL_TICKS = 15    # unknown screen before purchase -> drop round flags


class ForgeBotStateMachine:
    """
    Pure decision layer:
    - detects the current UI state from OCR text
    - decides the next high-level action
    - does not press keys itself
    """

    def __init__(
        self,
        relic_tokens: list[str],
        flatstone_tokens: list[str],
        main_menu_tokens: list[str],
        reset_menu_tokens: list[str],
        min_text_len: int,
        batch_size: int,
    ):
        self.relic_tokens = [t.lower() for t in relic_tokens]
        self.flatstone_tokens = [t.lower() for t in flatstone_tokens]
        self.main_menu_tokens = [t.lower() for t in main_menu_tokens]
        self.reset_menu_tokens = [t.lower() for t in reset_menu_tokens]
        self.min_text_len = min_text_len
        self.batch_size = batch_size


    def _menu_score(self, text: str, tokens: list[str]) -> int:
        if not text:
            return 0
        t = text.lower()
        return sum(1 for tok in tokens if tok and tok in t)

    def _is_relic_menu(self, tick: TickInput) -> bool:
        return tick.relic_token_hits >= 2

    def _is_flatstone_menu(self, tick: TickInput) -> bool:
        return tick.flatstone_token_hits >= 2

    def _is_main_menu(self, tick: TickInput) -> bool:
        return tick.main_menu_token_hits >= 2
    
    def _is_reset_menu(self, tick: TickInput) -> bool:
        return (
            tick.reset_token_hits >= 1 and tick.relic_token_hits < 2 and tick.flatstone_token_hits < 2 and tick.main_menu_token_hits < 2
            or tick.reset_token_hits >= 1 and len(tick.recognized_text) < tick.min_text_len
        )

    def is_relic_menu(self, text: str) -> bool:
        tick = self.detect_state(text, FlowContext())
        return self._is_relic_menu(tick)

    def is_flatstone_menu(self, text: str) -> bool:
        tick = self.detect_state(text, FlowContext())
        return self._is_flatstone_menu(tick)

    def is_main_menu(self, text: str) -> bool:
        tick = self.detect_state(text, FlowContext())
        return self._is_main_menu(tick)
    
    def is_reset_menu(self, text: str) -> bool:
        tick = self.detect_state(text, FlowContext())
        return self._is_reset_menu(tick)
    
    def is_batch_selected(self, text: str) -> bool:
        # "10/10" devient "1010" après fuzzy_clean_text
        # "1/10" devient "110"
        return "1010" in text

    def _count_hits(self, text: str, tokens: list[str]) -> int:
        if not text:
            return 0
        t = text.lower()
        return sum(1 for tok in tokens if tok in t)

#endregion
# region menu classification rules

    def detect_state(self, recognized_text: str, ctx: FlowContext) -> TickInput:
        text = (recognized_text or "").strip().lower()

        relic_hits = self._count_hits(text, self.relic_tokens)
        flatstone_hits = self._count_hits(text, self.flatstone_tokens)
        main_menu_hits = self._count_hits(text, self.main_menu_tokens)
        reset_menu_hits = self._count_hits(text, self.reset_menu_tokens)

        return TickInput(
            recognized_text=text,
            min_text_len=self.min_text_len,
            relic_token_hits=relic_hits,
            flatstone_token_hits=flatstone_hits,
            main_menu_token_hits=main_menu_hits,
            reset_token_hits=reset_menu_hits,
        )

    def classify_state(self, tick: TickInput, ctx: FlowContext) -> UIState:
        text = tick.recognized_text
        relic_hits = tick.relic_token_hits
        flatstone_hits = tick.flatstone_token_hits
        main_hits = tick.main_menu_token_hits

        # 1. Texte très faible : on reste prudent
        if not text or len(text) < tick.min_text_len:
            if ctx.transition_started:
                return UIState.TRANSITION
            if self._is_reset_menu(tick):
                return UIState.RESET_MENU
            return UIState.UNKNOWN

        # 2. Priorités de menus forts
        if relic_hits >= 2 and relic_hits > flatstone_hits and relic_hits > main_hits:
            return UIState.RELIC_MENU

        if flatstone_hits >= 2 and flatstone_hits > relic_hits and flatstone_hits > main_hits:
            return UIState.FLATSTONE_MENU

        if self._is_reset_menu(tick) and relic_hits < 2:
            return UIState.RESET_MENU

        if ctx.main_menu_grace_ticks > 0:
            if relic_hits < 2 and flatstone_hits < 2:
                return UIState.MAIN_MENU

        # 3. MAIN_MENU plus tolérant après batch terminé
        if ctx.processed_relics >= ctx.batch_size:
            if main_hits >= 1 and relic_hits < 2 and flatstone_hits < 2:
                return UIState.MAIN_MENU
            if relic_hits < 2 and flatstone_hits < 2:
                return UIState.MAIN_MENU
        else:
            if main_hits >= 2 and relic_hits < 2 and flatstone_hits < 2:
                return UIState.MAIN_MENU

        # 4. Priorité de transition
        if ctx.transition_started:
            return UIState.TRANSITION

        if self._is_reset_menu(tick):
            return UIState.RESET_MENU

        return UIState.UNKNOWN

# endregion
#region Decision logic

    def decide(self, recognized_text: str, ctx: FlowContext) -> TickDecision:
        tick = self.detect_state(recognized_text, ctx)
        state = self.classify_state(tick, ctx)

        updates: Dict[str, Any] = {
            "last_state": state,
            "stall_ticks": 0,
        }

        # Track empty OCR
        if not tick.recognized_text or len(tick.recognized_text) < tick.min_text_len:
            updates["empty_ticks"] = ctx.empty_ticks + 1
        else:
            updates["empty_ticks"] = 0

        if ctx.main_menu_grace_ticks > 0:
            updates["main_menu_grace_ticks"] = ctx.main_menu_grace_ticks - 1

        # Track global post-batch timeout once exit was initiated
        if (
            ctx.processed_relics >= ctx.batch_size
            and ctx.batch_exit_sent
            and state != UIState.MAIN_MENU
        ):
            updates["transition_ticks"] = ctx.transition_ticks + 1
        elif state == UIState.TRANSITION:
            updates["transition_ticks"] = ctx.transition_ticks + 1
        else:
            updates["transition_ticks"] = 0

        # ==========================================================
        # PRIORITÉ 1 : batch terminé
        # ==========================================================
        if ctx.processed_relics >= ctx.batch_size:
            # Retour stable au menu principal -> reset propre du round
            if state == UIState.MAIN_MENU:
                updates["batch_exit_sent"] = False
                updates["pending_new_round"] = True
                updates["round_active"] = False
                updates["transition_started"] = False
                updates["skip_sent"] = False
                updates["recovering"] = False
                updates["transition_ticks"] = 0
                updates["main_menu_grace_ticks"] = 8
                return TickDecision(
                    state=state,
                    action=Action.RESET_ROUND,
                    reason="Main menu reached after completed batch, reset before next round",
                    updates=updates,
                )

            # Si on voit un écran de reset après batch, on autorise une sortie
            if state == UIState.RESET_MENU:
                return TickDecision(
                    state=state,
                    action=Action.EXIT_RELIC_MENU,
                    reason="Reset menu detected after batch complete, exit with interact",
                    updates=updates,
                )

            # Si on voit encore une relique après batch complet, retenter la sortie
            if state == UIState.RELIC_MENU:
                if ctx.last_action != Action.EXIT_RELIC_MENU:
                    return TickDecision(
                        state=state,
                        action=Action.EXIT_RELIC_MENU,
                        reason="Relic menu still visible after batch complete, retry exit once",
                        updates=updates,
                    )
                return TickDecision(
                    state=state,
                    action=Action.WAIT,
                    reason="Relic menu still visible after batch complete, waiting briefly before retry",
                    updates=updates,
                )

            # Première sortie de fin de batch -> une seule fois
            if not ctx.batch_exit_sent:
                updates["batch_exit_sent"] = True
                updates["transition_started"] = True
                updates["skip_sent"] = False
                updates["transition_ticks"] = 0
                return TickDecision(
                    state=state,
                    action=Action.EXIT_RELIC_MENU,
                    reason="Batch complete, send exit only once",
                    updates=updates,
                )

            # Si l'UI reste hors MAIN_MENU trop longtemps après la sortie, on force le reset
            if ctx.batch_exit_sent and ctx.transition_ticks >= 12:
                updates["batch_exit_sent"] = False
                updates["pending_new_round"] = True
                updates["round_active"] = False
                updates["transition_started"] = False
                updates["skip_sent"] = False
                updates["recovering"] = False
                updates["transition_ticks"] = 0
                updates["main_menu_grace_ticks"] = 8
                return TickDecision(
                    state=UIState.MAIN_MENU,
                    action=Action.RESET_ROUND,
                    reason="Post-batch timeout reached outside main menu, force reset from presumed main menu",
                    updates=updates,
                )

            if state == UIState.TRANSITION:
                return TickDecision(
                    state=state,
                    action=Action.WAIT,
                    reason="Batch complete, waiting for main menu or reset menu",
                    updates=updates,
                )

            # Très important : si on voit FLATSTONE_MENU post-batch,
            # ne pas renvoyer interact de sortie
            if state == UIState.FLATSTONE_MENU:
                return TickDecision(
                    state=state,
                    action=Action.WAIT,
                    reason="Flatstone menu seen after batch complete, do not send exit interact here",
                    updates=updates,
                )

            # Si on est dans UNKNOWN après batch complet, ne pas recover vers relic menu
            if state == UIState.UNKNOWN:
                if ctx.empty_ticks >= 6:
                    updates["batch_exit_sent"] = False
                    updates["pending_new_round"] = True
                    updates["round_active"] = False
                    updates["transition_started"] = False
                    updates["skip_sent"] = False
                    updates["recovering"] = False
                    updates["transition_ticks"] = 0
                    updates["main_menu_grace_ticks"] = 8
                    return TickDecision(
                        state=UIState.MAIN_MENU,
                        action=Action.RESET_ROUND,
                        reason="Unknown post-batch state persisted, force reset from presumed main menu",
                        updates=updates,
                    )

                return TickDecision(
                    state=state,
                    action=Action.WAIT,
                    reason="Batch complete, unknown post-exit state, waiting for stable menu",
                    updates=updates,
                )

            # Sinon on attend un état stable
            return TickDecision(
                state=state,
                action=Action.WAIT,
                reason="Batch complete, waiting for stable post-batch state",
                updates=updates,
            )

        # ==========================================================
        # RELIC MENU
        # ==========================================================
        if state == UIState.RELIC_MENU:
            updates["transition_started"] = False
            updates["skip_sent"] = False
            updates["recovering"] = False
            updates["round_active"] = True
            updates["transition_ticks"] = 0

            return TickDecision(
                state=state,
                action=Action.PROCESS_CURRENT_RELIC,
                reason="Relic menu detected, process current relic",
                updates=updates,
            )

        # ==========================================================
        # FLATSTONE MENU
        # ==========================================================
        if state == UIState.FLATSTONE_MENU:
            updates["round_active"] = True
            updates["transition_started"] = False
            updates["skip_sent"] = False
            updates["recovering"] = False
            updates["transition_ticks"] = 0

            text = tick.recognized_text or ""
            batch_10_visible = "1010" in text

            # 1) Envoyer F2 une seule fois
            if not ctx.batch_select_requested:
                updates["batch_select_requested"] = True
                return TickDecision(
                    state=state,
                    action=Action.CHOOSE_BATCH_SIZE,
                    reason="Flatstone menu detected, send F2 once to request batch 10",
                    updates=updates,
                )

            # 2) Attendre confirmation visuelle du 10/10
            if not ctx.choose_10_done:
                if batch_10_visible:
                    # 10/10 is on screen in this very frame: confirm right away
                    # (falls through to step 3 instead of spending one more tick)
                    updates["choose_10_done"] = True
                else:
                    stall = ctx.stall_ticks + 1
                    if stall >= F2_RETRY_TICKS:
                        # F2 probably not registered: allow sending it again
                        # (ForgeFlow still spaces F2 presses by f2_retry_delay)
                        updates["batch_select_requested"] = False
                        return TickDecision(
                            state=state,
                            action=Action.WAIT,
                            reason=f"10/10 not seen after {stall} ticks, will resend F2",
                            updates=updates,
                        )
                    updates["stall_ticks"] = stall
                    return TickDecision(
                        state=state,
                        action=Action.WAIT,
                        reason="Waiting for visual confirmation of batch 10 after F2",
                        updates=updates,
                    )

            # 3) Confirmer l'achat
            if not ctx.confirm_done:
                updates["confirm_done"] = True
                updates["transition_started"] = True
                updates["skip_sent"] = False
                return TickDecision(
                    state=state,
                    action=Action.CONFIRM_FLATSTONE,
                    reason="Batch 10 confirmed visually, confirm flatstone purchase",
                    updates=updates,
                )

            # 4) Purchase screen still visible after confirm: never press F on it
            #    right away (it could buy again). ForgeFlow re-sends the confirm
            #    only if the screen stays frozen for confirm_retry_delay seconds.
            updates["transition_started"] = True
            return TickDecision(
                state=state,
                action=Action.RETRY_CONFIRM,
                reason="Still on purchase screen after confirm, wait (time-guarded confirm retry)",
                updates=updates,
            )

        # ==========================================================
        # TRANSITION
        # ==========================================================
        if state == UIState.TRANSITION:
            updates["round_active"] = True

            if not ctx.skip_sent:
                updates["skip_sent"] = True
                return TickDecision(
                    state=state,
                    action=Action.SKIP_TO_RELIC_MENU,
                    reason="Transition detected, send one skip input",
                    updates=updates,
                )

            return TickDecision(
                state=state,
                action=Action.RETRY_SKIP,
                reason="Transition in progress, wait (time-guarded skip retry)",
                updates=updates,
            )

        # ==========================================================
        # RESET MENU
        # ==========================================================
        if state == UIState.RESET_MENU:
            return TickDecision(
                state=state,
                action=Action.EXIT_RELIC_MENU,
                reason="Reset menu detected, exit with interact",
                updates=updates,
            )

        # ==========================================================
        # MAIN MENU
        # ==========================================================
        if state == UIState.MAIN_MENU:
            updates["recovering"] = False
            updates["transition_ticks"] = 0
            # Right after the purchase the game briefly shows the flatstone
            # description ("...purchase to reveal its precious gift"), which reads
            # as the main menu. Keep the transition flags: the black screen that
            # follows is the opening animation and must be skipped at once.
            if not (ctx.confirm_done and ctx.processed_relics == 0):
                updates["transition_started"] = False
                updates["skip_sent"] = False

            if ctx.pending_new_round:
                updates["pending_new_round"] = False
                updates["round_active"] = True
                return TickDecision(
                    state=state,
                    action=Action.ENTER_FLATSTONE,
                    reason="Confirmed main menu after reset, start next round",
                    updates=updates,
                )

            if not ctx.round_active:
                updates["round_active"] = True
                return TickDecision(
                    state=state,
                    action=Action.ENTER_FLATSTONE,
                    reason="Confirmed main menu, start first cycle",
                    updates=updates,
                )

            # Strong main-menu evidence (not grace-based) while a round is active
            # and no batch completed: the round was lost. Restart it after a while.
            strong_main = (
                tick.main_menu_token_hits >= 2
                and tick.flatstone_token_hits == 0
                and tick.relic_token_hits < 2
            )
            if strong_main:
                stall = ctx.stall_ticks + 1
                if stall >= MAIN_MENU_STALL_TICKS:
                    updates["pending_new_round"] = True
                    updates["round_active"] = False
                    return TickDecision(
                        state=state,
                        action=Action.RESET_ROUND,
                        reason=f"Main menu stable for {stall} ticks during active round, restart round",
                        updates=updates,
                    )
                updates["stall_ticks"] = stall

            return TickDecision(
                state=state,
                action=Action.WAIT,
                reason="Confirmed main menu during active round, wait",
                updates=updates,
            )

        # ==========================================================
        # UNKNOWN
        # ==========================================================
        purchase_done = ctx.confirm_done or ctx.processed_relics > 0
        if ctx.round_active and not purchase_done:
            # Before purchase, recovery presses F, which could buy a single
            # flatstone in the purchase menu. Wait instead, bounded.
            stall = ctx.stall_ticks + 1
            if stall >= UNKNOWN_STALL_TICKS:
                updates["pending_new_round"] = True
                updates["round_active"] = False
                return TickDecision(
                    state=state,
                    action=Action.RESET_ROUND,
                    reason=f"Unknown for {stall} ticks before purchase, reset and wait for main menu",
                    updates=updates,
                )
            updates["stall_ticks"] = stall
            return TickDecision(
                state=state,
                action=Action.WAIT,
                reason="Unknown state before purchase, wait (no recovery key)",
                updates=updates,
            )

        if ctx.round_active and ctx.processed_relics < ctx.batch_size:
            updates["recovering"] = True
            return TickDecision(
                state=state,
                action=Action.RECOVER_TO_RELIC_MENU,
                reason="Unknown state during active round, try recovering to relic menu",
                updates=updates,
            )

        return TickDecision(
            state=state,
            action=Action.WAIT,
            reason="Unknown idle state, wait",
            updates=updates,
        )

    @staticmethod
    def apply_updates(ctx: FlowContext, decision: TickDecision) -> None:
        for key, value in decision.updates.items():
            setattr(ctx, key, value)
        ctx.last_action = decision.action

# endregion