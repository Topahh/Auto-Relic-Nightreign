# utils/stats.py
# Session statistics tracker.
# Records kept items, blacklist-vetoed items, partial matches and scan counters
# across forging rounds.
# Outputs a formatted report to stdout and optionally writes it to a log file.

from __future__ import annotations
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from config.settings import Language
from utils.text import fuzzy_clean_text
from utils import log

if TYPE_CHECKING:
    from engine.matcher import MatchResult


# ---------------------------------------------------------------------------
# RelicRecord — structure légère flushée sur disque (~80–120 bytes JSON)
# ---------------------------------------------------------------------------

@dataclass
class RelicRecord:
    """
    Données minimales d'une relique pour les stats riches.
    On stocke les scores par groupe, pas le texte OCR.
    
    profiles: { "E1": {"a": 1, "a_min": 2, "b": 0, "b_req": False, "bl": 0} }
    """
    batch:   int
    index:   int
    outcome: str    # "kept" | "discard" | "partial" | "blacklisted"
    group:   str    # groupe validé ou best_partial group_name
    profiles: dict  # voir ci-dessus
    matched_kw: list

    def to_json_line(self) -> str:
        return json.dumps({
            "b": self.batch, "i": self.index,
            "o": self.outcome, "g": self.group,
            "p": self.profiles, "k": self.matched_kw,
        })

    @staticmethod
    def from_dict(d: dict) -> "RelicRecord":
        return RelicRecord(
            batch=d["b"], index=d["i"],
            outcome=d["o"], group=d["g"],
            profiles=d["p"], matched_kw=d["k"],
        )


def _profiles_from_results(all_results: Dict[str, "MatchResult"]) -> dict:
    out = {}
    for gname, r in all_results.items():
        if r.has_any_primary or r.keep or len(r.b_matched) > 0:  # ← ajouter "or len(r.b_matched) > 0"
            out[gname] = {
                "a":     len(r.a_matched),
                "a_min": r.a_min,
                "b":     len(r.b_matched),
                "b_req": r.b_required,
                "bl":    len(r.blacklist_hit),
                "keep":  r.keep,
            }
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class Statistics:
    """
    Tracks and reports item scanning statistics for a forging session.

    Counters (toujours en RAM, négligeable):
        rounds, scanned, kept

    Listes détaillées OCR (bornées par max_in_memory_details):
        kept_items, qualified_but_blacklisted, partial_matches

    Système de flush léger (NOUVEAU):
        _record_buffer  → vidé vers temp_path toutes les flush_every reliques
        Le fichier .jsonl est lu UNE SEULE FOIS en fin de session pour le rapport riche.
    """

    def __init__(
        self,
        lang: Language,
        max_in_memory_details: int = 300,
        flush_every: int = 300,
        temp_path: str = "/tmp/hajiwo_stats.jsonl",
    ):
        self.lang = lang
        self.rounds = 0
        self.scanned = 0
        self.kept = 0

        # Compteurs cumulés (jamais vidés)
        self.blacklisted = 0
        self.partial = 0
        self.kept_by_group: Dict[str, int] = defaultdict(int)
        self.blacklist_reasons: Dict[str, int] = defaultdict(int)
        self.kept_summary: List[str] = []   # "[E1] kw1, kw2" — toute la session
        self._batch_start_scanned = 0
        self._batch_start_kept = 0

        self.kept_items: list = []
        self.qualified_but_blacklisted: list = []
        self.partial_matches: list = []

        self.max_in_memory_details = max_in_memory_details
        self.flush_count = 0

        # ── Nouveau système de flush ──────────────────────────────────
        self.flush_every   = flush_every
        self.temp_path     = temp_path
        self._record_buffer: List[RelicRecord] = []
        self._records_written = 0
        # temp_path is in /tmp = RAM (tmpfs): cap it for weeks-long sessions
        self.max_records = 200_000      # ~20 MB
        self._records_capped = False

        # Drop leftovers from a previous session that ended without cleanup,
        # otherwise they would be counted in this session's summary.
        try:
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)
        except OSError as e:
            print(f"[STATS][WARN] Could not remove stale temp file: {e}")

    # ------------------------------------------------------------------
    # Helpers listes OCR (inchangé)
    # ------------------------------------------------------------------

    def _trim_list(self, items: list):
        overflow = len(items) - self.max_in_memory_details
        if overflow > 0:
            del items[:overflow]

    def _trim_all(self):
        self._trim_list(self.kept_items)
        self._trim_list(self.qualified_but_blacklisted)
        self._trim_list(self.partial_matches)

    def clear_detailed_lists(self):
        self.kept_items.clear()
        self.qualified_but_blacklisted.clear()
        self.partial_matches.clear()

    # ------------------------------------------------------------------
    # Enregistrement classique (inchangé)
    # ------------------------------------------------------------------

    def add_kept_item(self, texts, keywords, group_name):
        self.kept_items.append((texts, keywords, group_name))
        self._trim_list(self.kept_items)
        self.kept_by_group[group_name or "?"] += 1
        self.kept_summary.append(f"[{group_name}] {', '.join(keywords)}")
        self._trim_list(self.kept_summary)

    def add_qualified_blacklisted(self, texts, matched_keywords, blacklist_keywords):
        self.blacklisted += 1
        for kw in blacklist_keywords:
            self.blacklist_reasons[kw] += 1
        self.qualified_but_blacklisted.append((texts, matched_keywords, blacklist_keywords))
        self._trim_list(self.qualified_but_blacklisted)

    def add_partial_match(self, texts, matched_keywords, reason, group_name):
        self.partial += 1
        self.partial_matches.append((texts, matched_keywords, reason, group_name))
        self._trim_list(self.partial_matches)

    def batch_line(self) -> str:
        """Une ligne de résumé à la fin de chaque lot de reliques."""
        scanned = self.scanned - self._batch_start_scanned
        kept = self.kept - self._batch_start_kept
        self._batch_start_scanned = self.scanned
        self._batch_start_kept = self.kept
        return (f"── Batch {self.rounds} : {kept}/{scanned} kept"
                f"  | total {self.kept}/{self.scanned} ──")

    # ------------------------------------------------------------------
    # NOUVEAU — enregistrement structuré léger
    # ------------------------------------------------------------------

    def record_relic(
        self,
        batch: int,
        index: int,
        outcome: str,
        group: str,
        all_results: Dict[str, "MatchResult"],
        matched_kw: list,
    ):
        """
        Buffeurise une RelicRecord légère. Flush auto toutes les flush_every reliques.
        
        Appeler après chaque décision keep/discard dans process_item().
        all_results vient du 7e élément de matcher.match_detailed().
        """
        if self._records_written + len(self._record_buffer) >= self.max_records:
            if not self._records_capped:
                self._records_capped = True
                print(f"[STATS] rich_stats: {self.max_records:,} relics recorded, "
                      f"recording stopped to protect RAM (/tmp)")
            return
        profiles = _profiles_from_results(all_results)
        rec = RelicRecord(
            batch=batch, index=index, outcome=outcome,
            group=group, profiles=profiles,
            matched_kw=matched_kw[:10],  # cap pour éviter les keywords très longues
        )
        self._record_buffer.append(rec)

        if len(self._record_buffer) >= self.flush_every:
            self._flush_records()

    def _flush_records(self):
        """Écrit le buffer courant en JSON Lines et le vide."""
        if not self._record_buffer:
            return
        try:
            with open(self.temp_path, 'a', encoding='utf-8') as f:
                for rec in self._record_buffer:
                    f.write(rec.to_json_line() + "\n")
            self._records_written += len(self._record_buffer)
            log.dbg(
                f"[STATS] Flushed {len(self._record_buffer)} records "
                f"(total on disk={self._records_written}) → {self.temp_path}"
            )
        except Exception as e:
            print(f"[STATS][ERROR] Flush failed: {e}")
        finally:
            self._record_buffer.clear()

    def flush_final(self):
        """Force-flush les enregistrements restants. Appeler avant la fin du programme."""
        self._flush_records()

    def cleanup_temp(self):
        """Supprime le fichier .jsonl temporaire après écriture du rapport final."""
        try:
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)
                log.dbg(f"[STATS] Temp file deleted: {self.temp_path}")
        except Exception as e:
            print(f"[STATS][ERROR] Cleanup failed: {e}")

    # ------------------------------------------------------------------
    # NOUVEAU — synthèse riche (lue depuis disque, zéro RAM supplémentaire)
    # ------------------------------------------------------------------

    def _build_rich_summary(self) -> List[str]:
        """
        Lit le .jsonl et calcule les agrégats de session.
        N'est appelé QU'UNE FOIS en fin de programme.
        Complexité mémoire : O(nb_groupes * nb_profils), pas O(nb_reliques).
        """
        # S'assurer que le buffer résiduel est bien flushé
        if self._record_buffer:
            self._flush_records()

        if not os.path.exists(self.temp_path):
            return ["[RICH STATS] No temp data file found — nothing was recorded."]

        # ── Structures d'agrégation ────────────────────────────────────
        total = 0
        outcomes: dict[str, int] = defaultdict(int)
        
        # Par groupe : { group_name: { "a_depth": [0,0,...], "b_hit": int, "bl_hit": int,
        #                              "outcomes": {outcome: count} } }
        group_stats: dict[str, dict] = {}
        
        # Combos de groupes partiellement matchés simultanément
        combo_counts: dict[str, int] = defaultdict(int)
        
        # Distribution des scores A pour chaque groupe (indépendamment)
        # profile_depth[group][a_score] = count
        profile_depth: dict[str, list] = {}

        try:
            with open(self.temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    total += 1
                    outcome = d.get("o", "?")
                    group   = d.get("g", "") or "—"
                    profiles: dict = d.get("p", {})

                    outcomes[outcome] += 1

                    # Profondeur de matching par groupe
                    for gname, pdata in profiles.items():
                        a_score = pdata.get("a", 0)
                        a_min   = pdata.get("a_min", 1)
                        b_hit   = pdata.get("b", 0)
                        bl_hit  = pdata.get("bl", 0)
                        kept    = pdata.get("keep", False)

                        # Init profil si nouveau
                        if gname not in profile_depth:
                            profile_depth[gname] = [0] * (a_min + 1)
                        arr = profile_depth[gname]
                        while len(arr) <= a_score:
                            arr.append(0)
                        arr[a_score] += 1

                        # Agrégats groupe
                        if gname not in group_stats:
                            group_stats[gname] = {
                                "total": 0, "b_hit_passed": 0, "bl_hit": 0,
                                "kept": 0, "a_min": a_min, "b_req": pdata.get("b_req", False),  # ← b_req ajouté
                                "outcomes": defaultdict(int),
                            }
                        gs = group_stats[gname]
                        gs["total"]   += 1
                        gs["bl_hit"]  += 1 if bl_hit else 0
                        gs["kept"]    += 1 if kept else 0
                        gs["outcomes"][outcome] += 1

                        # b_hit compté SEULEMENT parmi les reliques qui ont passé le seuil A
                        if a_score >= a_min:
                            gs["b_hit_passed"] = gs.get("b_hit_passed", 0) + (1 if b_hit else 0)  # ← remplace gs["b_hit"]

                    # Combos : groupes avec a_score > 0 simultanément sur la même relique
                    hit_groups = sorted(
                        gname for gname, pd in profiles.items()
                        if pd.get("a", 0) > 0
                    )
                    if len(hit_groups) >= 2:
                        combo_counts[" + ".join(hit_groups)] += 1

        except Exception as e:
            return [f"[RICH STATS][ERROR] Failed to read temp file: {e}"]

        if total == 0:
            return ["[RICH STATS] 0 records found in temp file."]

        # ── Formatage ─────────────────────────────────────────────────
        W = 50
        lines = []
        lines.append("=" * W)
        lines.append(" RICH STATISTICS — Full Session Analysis")
        lines.append("=" * W)
        lines.append(f"  Total relics analyzed : {total:,}")
        lines.append(f"  Batches completed     : {self.rounds}")
        lines.append("")

        # Outcome breakdown
        lines.append("── Outcome breakdown " + "─" * 29)
        for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
            bar = "█" * int(v / total * 30)
            pct = v / total * 100
            lines.append(f"  {k:<14} {v:>7,}  {pct:5.1f}%  {bar}")
        lines.append("")

        # Profile depth — score A atteint par groupe
        lines.append("── Profile matching depth (A score distribution) " + "─" * 1)
        lines.append("   Shows how far each group's primary keywords were matched")
        lines.append("")
        for gname in sorted(profile_depth.keys()):
            arr   = profile_depth[gname]
            gs    = group_stats.get(gname, {})
            a_min = gs.get("a_min", 1)
            b_req = gs.get("b_req", False)

            if b_req:
                lines.append(f"  [{gname}]  (need {a_min} from pool_a  +  1 required optional)")
            else:
                lines.append(f"  [{gname}]  (need {a_min} from pool_a)")

            # Compter les reliques ayant passé le seuil A
            passed_count = sum(cnt for score, cnt in enumerate(arr) if score >= a_min)

            # Afficher les lignes score < a_min normalement
            for score, cnt in enumerate(arr):
                if cnt == 0:
                    continue
                if score >= a_min:
                    break  # on affiche la ligne agrégée après
                pct = cnt / total * 100
                bar = "·" * int(pct / 2)
                lines.append(f"    {score}/{len(arr)-1} : {cnt:>7,}  ({pct:5.1f}%)  {bar}")

            # Ligne agrégée ≥a_min
            if passed_count > 0:
                pct = passed_count / total * 100
                bar = "·" * int(pct / 2)
                lines.append(f"    ≥{a_min}/{len(arr)-1}: {passed_count:>7,}  ({pct:5.1f}%)  {bar}  ← A threshold met")

                if b_req:
                    b_hit_passed = gs.get("b_hit_passed", 0)
                    no_b         = passed_count - b_hit_passed
                    b_pct        = b_hit_passed / passed_count * 100 if passed_count else 0
                    no_b_pct     = no_b / passed_count * 100 if passed_count else 0
                    lines.append(f"      └─ with b : {b_hit_passed:>6,}  ({b_pct:5.1f}%)  ← FULL PASS")
                    lines.append(f"      └─ no  b : {no_b:>6,}  ({no_b_pct:5.1f}%)  ← missing required optional")

            # Blacklist summary
            bl_hit = gs.get("bl_hit", 0)
            kept   = gs.get("kept", 0)
            if bl_hit or kept:
                details = []
                if bl_hit: details.append(f"blacklisted={bl_hit:,}")
                if kept:   details.append(f"fully_kept={kept:,}")
                lines.append(f"    └─ {' | '.join(details)}")
            lines.append("")

        # Combos multi-groupes
        if combo_counts:
            lines.append("── Multi-group partial hits (same relic) " + "─" * 9)
            lines.append("   Relics that partially matched multiple groups simultaneously")
            sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:10]
            for combo, cnt in sorted_combos:
                pct = cnt / total * 100
                lines.append(f"  {combo:<36}  {cnt:>6,}  ({pct:5.1f}%)")
            lines.append("")

        # Top matched keywords (parmi les reliques partiellement matchées)
        # On compte les kw depuis les records avec outcome partial/blacklisted
        # (Pour ça on refait un mini-pass — c'est acceptable car on est en fin de session)
        kw_counts: dict[str, int] = defaultdict(int)
        try:
            with open(self.temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("o") in ("partial", "blacklisted"):
                        for kw in d.get("k", []):
                            kw_counts[kw] += 1
        except Exception:
            pass

        if kw_counts:
            lines.append("── Top partial-match keywords (partial/blacklisted relics) ─")
            sorted_kw = sorted(kw_counts.items(), key=lambda x: -x[1])[:15]
            for kw, cnt in sorted_kw:
                pct = cnt / total * 100
                lines.append(f"  {kw[:40]:<42}  {cnt:>6,}  ({pct:5.1f}%)")
            lines.append("")

        lines.append("=" * W)
        return lines

    def print_rich_summary(self):
        """Affiche le rapport riche sur stdout."""
        print("\n" + "\n".join(self._build_rich_summary()))

    def save_rich_summary(self, filepath: str):
        """Append le rapport riche dans le fichier de log final."""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write("\n".join(self._build_rich_summary()) + "\n\n")
            log.dbg(f"[STATS] Rich summary appended to: {filepath}")
        except Exception as e:
            print(f"[STATS][ERROR] Failed to write rich summary: {e}")

    # ------------------------------------------------------------------
    # Rapport classique (inchangé)
    # ------------------------------------------------------------------

    def _build_report_lines(
        self,
        include_details: bool = True,
        header_suffix: Optional[str] = None,
        final: bool = False,
    ) -> List[str]:
        """Rapport compact. Les listes partial / blacklisted détaillées
        ne sont écrites qu'en mode debug ([Logging] debug = true)."""
        lang = self.lang
        title = lang.get('stats_title')
        if header_suffix:
            title = f"{title} - {header_suffix}"
        rate = (self.kept / self.scanned * 100) if self.scanned else 0.0
        lines = [
            f"── {title} ──",
            f"{lang.get('total_rounds')} {self.rounds} | {lang.get('total_scanned')} {self.scanned}"
            f" | {lang.get('total_kept')} {self.kept} ({rate:.1f}%)"
            f" | blacklist {self.blacklisted} | partial {self.partial}",
        ]
        if self.kept_by_group:
            lines.append("Kept by group: " + ", ".join(
                f"{g} {n}" for g, n in sorted(self.kept_by_group.items(), key=lambda x: -x[1])))
        if self.blacklist_reasons:
            top = sorted(self.blacklist_reasons.items(), key=lambda x: -x[1])[:5]
            lines.append("Top blacklist: " + ", ".join(f"{k} {n}" for k, n in top))

        if include_details:
            # Final : toute la session ; snapshot : seulement depuis le dernier snapshot
            kept_lines = self.kept_summary if final else [
                f"[{g}] {', '.join(kw)}" for _, kw, g in self.kept_items]
            if kept_lines:
                lines.append(f"{lang.get('kept_items')}:")
                lines += [f"  {i}. {k}" for i, k in enumerate(kept_lines, 1)]

            if log.DEBUG:
                for idx, (texts, matched_kw, reason, group_name) in enumerate(self.partial_matches, 1):
                    lines.append(f"  partial {idx}. [{group_name}] {reason}")
                for idx, (texts, matched_kw, blacklist_kw) in enumerate(self.qualified_but_blacklisted, 1):
                    lines.append(f"  blacklisted {idx}. {', '.join(matched_kw)} -> (({', '.join(blacklist_kw)}))")

        return lines

    def print_report(self):
        print("\n" + "\n".join(self._build_report_lines(header_suffix="session", final=True)))

    def save_log(self, filepath: str):
        lang = self.lang
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(self._build_report_lines()) + "\n")
            print(f"\n[OK] {lang.get('log_saved')}: {filepath}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save log: {e}")

    def append_log_snapshot(
        self,
        filepath: str,
        header_suffix: Optional[str] = None,
        include_details: bool = True,
        clear_after_write: bool = False,
        final: bool = False,
    ):
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(
                    "\n".join(self._build_report_lines(
                        include_details=include_details,
                        header_suffix=header_suffix,
                        final=final,
                    )) + "\n\n"
                )
            self.flush_count += 1
            log.dbg(f"[STATS] Snapshot appended to: {filepath}")
            if clear_after_write:
                self.clear_detailed_lists()
                log.dbg("[STATS] Detailed lists cleared from memory after flush")
        except Exception as e:
            print(f"\n[ERROR] Failed to append snapshot: {e}")