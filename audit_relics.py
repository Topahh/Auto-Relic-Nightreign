#!/usr/bin/env python3
"""Rejoue un fichier debug_captures/hajiwo_relics_*.tsv avec les groupes
ACTUELS de hajiwo.ini, pour chercher les faux negatifs.

Usage : python3 audit_relics.py [fichier.tsv]   (defaut : le plus recent)

Affiche :
  - les reliques jetees que le filtre actuel garderait (config modifiee depuis) ;
  - les reliques jetees "presque gardees" : un groupe dont _b est satisfait
    (ou vide) et a qui il ne manque qu'UN mot-cle _a, sans blacklist ;
  - les mots-cles approchants (1-2 caracteres de difference) = erreur OCR probable.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import Config          # noqa: E402
from engine.matcher import ItemMatcher      # noqa: E402
from utils.text import natural_sort_key     # noqa: E402


def near_distance(kw, text, k):
    """Plus petite distance d'edition entre kw et une sous-chaine de text."""
    prev = list(range(len(kw) + 1))
    best = len(kw)
    for ch in text:
        cur = [0]
        for j in range(1, len(kw) + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (kw[j - 1] != ch)))
        best = min(best, cur[-1])
        if best <= k:
            return best
        prev = cur
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "debug_captures", "hajiwo_relics_*.tsv")),
        key=os.path.getmtime, default=None)
    if not path:
        print("Aucun fichier hajiwo_relics_*.tsv trouve.")
        return 1

    cfg = Config()
    matcher = ItemMatcher(cfg)
    groups = cfg.KEYWORD_GROUPS
    all_kw = {kw for g in groups.values() for kw in g["a"] + g["b"]}

    rows = []
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 6:
                rows.append(parts)

    newly_kept, almost, ocr = [], [], []
    for batch, idx, outcome, group, bl, text in rows:
        if outcome == "kept":
            continue
        keep, info, *_ = matcher.match_detailed([text])
        if keep:
            newly_kept.append((batch, idx, info, text))
            continue
        for gname in sorted(groups, key=natural_sort_key):
            g = groups[gname]
            if any(k in text for k in g["blacklist"]):
                continue
            if g["b"] and not any(k in text for k in g["b"]):
                continue
            hits = [k for k in g["a"] if k in text]
            if len(hits) == g["min"] - 1:
                almost.append((batch, idx, gname, hits, text))
                break
        for kw in all_kw:
            if len(kw) >= 10 and kw not in text:
                if kw.rstrip("0123456789") in text:
                    continue  # autre palier (+2 au lieu de +4) : pas une erreur OCR
                k = 1 if len(kw) < 18 else 2
                if near_distance(kw, text, k) <= k:
                    ocr.append((batch, idx, kw, text))

    print(f"{path}\n{len(rows)} reliques, {sum(r[2] == 'kept' for r in rows)} gardees en session\n")
    print(f"== Jetees mais gardees par la config ACTUELLE : {len(newly_kept)}")
    for b, i, info, t in newly_kept:
        print(f"  lot {b} #{i}  {info}\n      {t}")
    print(f"\n== Presque gardees (il manque 1 mot-cle _a) : {len(almost)}")
    for b, i, g, hits, t in almost:
        print(f"  lot {b} #{i}  [{g}] trouve : {', '.join(hits) or '-'}\n      {t}")
    print(f"\n== Mots-cles approchants (erreur OCR probable) : {len(ocr)}")
    for b, i, kw, t in ocr:
        print(f"  lot {b} #{i}  ~{kw}\n      {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
