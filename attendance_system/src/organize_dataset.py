"""
Phase 2 — Dataset Organization (multi-session)
==============================================

Maintains the identity mapping (``data/labels/labels.csv``) across ALL
capture sessions and builds the class-labeled dataset tree from it.

Workflow
--------
1. ``python src/organize_dataset.py init --session <tag>``
       Appends one row per not-yet-listed image of that session.
       * First-ever run on the DSLR session (bootstrap): provisional
         sequential IDs (S01, S02, ...) are assigned in filename order for
         the researcher to verify against the contact sheet.
       * Ingested sessions (phone1, ...): the student id is parsed from the
         filename prefix written by ingest_session.py, so rows arrive
         pre-labeled (still marked verified=no until reviewed).
       Existing rows are never modified; ``--force`` regenerates the rows of
       the named session only (discarding manual edits for that session).

2. ``python src/organize_dataset.py build [--sessions dslr,phone1]``
       Validates the label file, then (re)builds::

           data/organized/<student_id>/<student_id>_<source>_<seq>.<ext>

       Files are byte-identical copies (SHA-256 verified). Without
       ``--sessions`` the whole tree is rebuilt; with it, only the named
       sessions' files are refreshed and other sessions' copies are kept —
       so adding phone data never touches the verified DSLR organization.

Design decisions
----------------
* Labels live in ONE auditable CSV rather than hand-sorted folders — manual
  file moving is the classic source of silent label noise.
* Student IDs are pseudonyms (S01...): no filename, figure, or log ever
  needs a participant's real name (ethics requirement).
* The ``source`` tag records the capture session, enabling session-disjoint
  train/test splits (deployment-realistic evaluation).
* Filenames are globally unique across sessions (see config.py invariant),
  so a bare filename is an unambiguous key in every metadata file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import shutil
import sys
from collections import defaultdict

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("organize")

FIELDS = ["filename", "student_id", "source", "student_name", "verified"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels() -> list[dict[str, str]]:
    if not config.LABELS_CSV.exists():
        return []
    with open(config.LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_labels(rows: list[dict[str, str]]) -> None:
    config.LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# subcommand: init
# ---------------------------------------------------------------------------
def cmd_init(session: str, force: bool) -> None:
    paths = config.session_images(session)
    if not paths:
        raise SystemExit(f"no images found for session '{session}' "
                         f"(dir: {config.SESSIONS[session]['images_dir']})")

    rows = read_labels()
    bootstrap = not rows
    if force:
        dropped = sum(1 for r in rows if r["source"] == session)
        rows = [r for r in rows if r["source"] != session]
        if dropped:
            log.warning("--force dropped %d existing '%s' rows "
                        "(manual edits to them are lost)", dropped, session)
    known = {r["filename"] for r in rows}

    sid_re = re.compile(rf"^{re.escape(session)}_(S\d{{2,3}})_")
    added = unparsed = 0
    seq = 1
    for p in paths:
        if p.name in known:
            continue
        m = sid_re.match(p.name)
        if m:
            sid = m.group(1)
        elif bootstrap and session == "dslr":
            sid = f"S{seq:02d}"       # provisional; verify via contact sheet
        else:
            sid = ""                  # must be filled before build validates
            unparsed += 1
        seq += 1
        rows.append(dict(zip(FIELDS, [p.name, sid, session, "", "no"])))
        added += 1

    write_labels(rows)
    log.info("labels.csv: +%d rows for session '%s' (total %d rows)",
             added, session, len(rows))
    if unparsed:
        log.warning("%d new rows have EMPTY student_id (filename prefix not "
                    "parseable) — fill them before running build", unparsed)
    log.info("NEXT: verify the new rows, set verified=yes, then run:  "
             "organize_dataset.py build")


# ---------------------------------------------------------------------------
# subcommand: build
# ---------------------------------------------------------------------------
def validate(rows: list[dict[str, str]],
             sessions: set[str]) -> list[str]:
    """Human-readable problems for the selected sessions ([] = valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):
        fn, sid, src = row["filename"], row["student_id"], row["source"]
        if src not in config.KNOWN_SOURCES:
            problems.append(f"line {i}: unknown source '{src}' "
                            f"(known: {sorted(config.KNOWN_SOURCES)})")
            continue
        if src not in sessions:
            continue
        if fn in seen:
            problems.append(f"line {i}: duplicate filename entry '{fn}'")
        seen.add(fn)
        if not config.find_image(src, fn).exists():
            problems.append(f"line {i}: file '{fn}' missing from session "
                            f"'{src}' directory")
        if not re.match(config.STUDENT_ID_PATTERN, sid):
            problems.append(f"line {i}: student_id '{sid}' does not match "
                            "S01-S999 (empty = not yet labeled)")
    for src in sessions:
        for p in config.session_images(src):
            if p.name not in seen:
                problems.append(f"image '{p.name}' (session '{src}') has no "
                                "row in labels.csv — run init --session "
                                f"{src}")
    return problems


def cmd_build(sessions_arg: str | None) -> None:
    all_rows = read_labels()
    if not all_rows:
        raise SystemExit("labels.csv is empty — run 'init' first.")
    sessions = (set(sessions_arg.split(",")) if sessions_arg
                else set(config.KNOWN_SOURCES))
    unknown = sessions - config.KNOWN_SOURCES
    if unknown:
        raise SystemExit(f"unknown session(s): {sorted(unknown)}")
    rows = [r for r in all_rows if r["source"] in sessions]
    if not rows:
        raise SystemExit(f"no labels rows for sessions {sorted(sessions)}")

    problems = validate(all_rows, sessions)
    if problems:
        for p in problems:
            log.error(p)
        raise SystemExit(f"label validation failed with {len(problems)} "
                         "problem(s)")

    unverified = sum(1 for r in rows if r.get("verified", "no").lower() != "yes")
    if unverified:
        log.warning("%d/%d selected rows still have verified=no — identities "
                    "are PROVISIONAL (limitation L3)", unverified, len(rows))

    # organized/ is derived. Full build: wipe everything. Partial build:
    # remove only the selected sessions' files, keep other sessions intact.
    if sessions == set(config.KNOWN_SOURCES):
        if config.ORGANIZED_DIR.exists():
            shutil.rmtree(config.ORGANIZED_DIR)
    else:
        for src in sessions:
            for f in config.ORGANIZED_DIR.glob(f"*/*_{src}_*"):
                f.unlink()
    config.ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["student_id"], row["source"])].append(row)

    n_copied = 0
    for (sid, src), members in sorted(groups.items()):
        class_dir = config.ORGANIZED_DIR / sid
        class_dir.mkdir(exist_ok=True)
        for seq, row in enumerate(sorted(members, key=lambda r: r["filename"]),
                                  start=1):
            src_path = config.find_image(src, row["filename"])
            dst_path = class_dir / f"{sid}_{src}_{seq:03d}{src_path.suffix.lower()}"
            shutil.copy2(src_path, dst_path)
            if sha256_file(src_path) != sha256_file(dst_path):
                raise SystemExit(f"integrity check FAILED for {dst_path}")
            n_copied += 1
    log.info("copied + SHA-256 verified %d files into %s",
             n_copied, config.ORGANIZED_DIR)

    # ---- class distribution over the ENTIRE label file ---------------------
    out_dir = config.REPORTS_DIR / "dataset_organization"
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_path = out_dir / "class_distribution.csv"
    per_class: dict[str, list[str]] = defaultdict(list)
    for row in all_rows:
        per_class[row["student_id"]].append(row["source"])
    with open(dist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "n_images", "sources"])
        for sid in sorted(per_class):
            srcs = per_class[sid]
            w.writerow([sid, len(srcs), "+".join(sorted(set(srcs)))])
    log.info("class distribution -> %s", dist_path)

    counts = sorted(len(v) for v in per_class.values())
    print("\n============ ORGANIZATION SUMMARY ============")
    print(f"sessions built       : {sorted(sessions)}")
    print(f"classes (students)   : {len(per_class)}")
    print(f"files copied         : {n_copied}")
    print(f"images per class     : min {counts[0]}, "
          f"median {counts[len(counts) // 2]}, max {counts[-1]}")
    print(f"unverified labels    : {unverified} (selected sessions)")
    print("==============================================")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init", help="append label rows for a session")
    p_init.add_argument("--session", default="dslr",
                        choices=sorted(config.KNOWN_SOURCES))
    p_init.add_argument("--force", action="store_true",
                        help="regenerate this session's rows (discards edits)")
    p_build = sub.add_parser("build",
                             help="validate labels and build data/organized")
    p_build.add_argument("--sessions",
                         help="comma list (default: all), e.g. dslr,phone1")
    args = ap.parse_args()

    if args.cmd == "init":
        cmd_init(session=args.session, force=args.force)
    elif args.cmd == "build":
        cmd_build(args.sessions)


if __name__ == "__main__":
    sys.exit(main())
