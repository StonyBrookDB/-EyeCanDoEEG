"""
P300 SPELLER EXPERIMENT — Emotiv EPOC Flex, 6x6 Farwell-Donchin
copy-spelling (OFFLINE calibration).

EEG is recorded manually in EmotivPRO (the Cortex API raw-EEG license is not
required). This script handles only the stimulus presentation and saves a
markers.csv with wall-clock timestamps (time.time() / UTC Unix seconds) that
align with the timestamps EmotivPRO embeds in its exported EDF/CSV file.

Workflow
--------
  1. Open EmotivPRO, check impedance, start a new recording — do NOT press
     Stop until the script prints "DONE".
  2. python p300_speller_experiment_emotiv.py --phrase HELLO --reps 15
  3. The script prints a countdown and then runs the paradigm.
  4. After it prints "DONE", stop the EmotivPRO recording and export as EDF.
  5. Run decode_emotivpro.py <session_dir> <edf_file> to align + decode.

Output (in recordings/session_<n>/)
--------------------------------------
  markers.csv   lsl_timestamp, wall_time, code, is_target, char_idx, rep
  meta.json     grid, timing, phrase, seed

Run (real experiment, GUI window):
    python p300_speller_experiment_emotiv.py --phrase HELLO --reps 15

Run (headless dry-run, no window):
    python p300_speller_experiment_emotiv.py --phrase AB --reps 3 --headless
"""
import argparse
import csv
import json
import os
import random
import time

from pylsl import StreamInfo, StreamOutlet, local_clock

# ---------------------------------------------------------------------------
# 6x6 grid
# ---------------------------------------------------------------------------
GRID = ["ABCDEF",
        "GHIJKL",
        "MNOPQR",
        "STUVWX",
        "YZ1234",
        "56789_"]
N = 6


def char_to_rc(ch):
    ch = ch.upper()
    for r, row in enumerate(GRID):
        c = row.find(ch)
        if c >= 0:
            return r, c
    raise ValueError(f"character {ch!r} is not on the 6x6 grid")


def char_to_target_codes(ch):
    r, c = char_to_rc(ch)
    return c + 1, r + 7


def code_pair_to_char(row_code, col_code):
    return GRID[row_code - 7][col_code - 1]


def cells_for_code(code):
    if 1 <= code <= 6:
        return [(r, code - 1) for r in range(N)]
    elif 7 <= code <= 12:
        return [(code - 7, c) for c in range(N)]
    raise ValueError(code)


def reconstruct_phrase(marker_rows):
    """Sanity-check: rebuild phrase from saved target markers."""
    per_char = {}
    for row in marker_rows:
        *_, code, is_target, char_idx, _ = row
        if int(is_target) == 1:
            per_char.setdefault(int(char_idx), set()).add(int(code))
    phrase = []
    for ci in sorted(per_char):
        codes = per_char[ci]
        cols = [c for c in codes if 1 <= c <= 6]
        rows = [c for c in codes if 7 <= c <= 12]
        phrase.append(code_pair_to_char(rows[0], cols[0]) if cols and rows else "?")
    return "".join(phrase)


# ---------------------------------------------------------------------------
# Flash schedule
# ---------------------------------------------------------------------------
def build_schedule(phrase, reps, rng):
    steps = []
    last_code = None
    for char_idx, ch in enumerate(phrase):
        col_code, row_code = char_to_target_codes(ch)
        targets = {col_code, row_code}
        steps.append({"type": "cue", "char_idx": char_idx, "target_char": ch})
        for rep in range(1, reps + 1):
            order = list(range(1, 13))
            rng.shuffle(order)
            if order[0] == last_code and len(order) > 1:
                order[0], order[1] = order[1], order[0]
            for code in order:
                steps.append({
                    "type": "flash",
                    "char_idx": char_idx,
                    "rep": rep,
                    "code": code,
                    "is_target": int(code in targets),
                })
                last_code = code
        steps.append({"type": "rest", "char_idx": char_idx, "target_char": ch})
    steps.append({"type": "end"})
    return steps


# ---------------------------------------------------------------------------
# Marker writer — saves markers to memory, flushes to CSV at the end
# ---------------------------------------------------------------------------
class MarkerRecorder:
    def __init__(self, outdir, marker_outlet):
        self.outdir = outdir
        self.marker_outlet = marker_outlet
        self.rows = []   # [lsl_ts, wall_time, code, is_target, char_idx, rep]

    def push(self, code, is_target, char_idx, rep):
        lsl_ts   = local_clock()
        wall_time = time.time()
        self.marker_outlet.push_sample(
            [float(code), float(is_target), float(char_idx), float(rep)]
        )
        self.rows.append([lsl_ts, wall_time, code, is_target, char_idx, rep])

    def flush(self):
        path = os.path.join(self.outdir, "markers.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lsl_timestamp", "wall_time", "code",
                        "is_target", "char_idx", "rep"])
            for row in self.rows:
                lsl_ts, wt, code, tgt, ci, rep = row
                w.writerow([lsl_ts, f"{wt:.6f}", int(code), int(tgt),
                             int(ci), int(rep)])
        print(f"[recorder] wrote {len(self.rows)} markers -> {path}")
        return path


# ---------------------------------------------------------------------------
# GUI presentation (Tkinter)
# ---------------------------------------------------------------------------
DIM_FG = "#5a5a5a"
ON_FG  = "#ffffff"
CUE_FG = "#34d058"
BG     = "#000000"


def run_gui(schedule, mrec, timing, on_start):
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("P300 6x6 Speller — Emotiv EPOC Flex")
    root.configure(bg=BG)
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
    root.bind("<F11>",
              lambda e: root.attributes("-fullscreen",
                                        not root.attributes("-fullscreen")))

    big         = tkfont.Font(family="Consolas", size=72, weight="bold")
    status_font = tkfont.Font(family="Consolas", size=24)

    status = tk.Label(root, text="", fg="#cccccc", bg=BG, font=status_font)
    status.pack(pady=(28, 10))

    board = tk.Frame(root, bg=BG)
    board.pack(expand=True)
    cells = [[None] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            lbl = tk.Label(board, text=GRID[r][c], fg=DIM_FG, bg=BG,
                           font=big, width=2, padx=10, pady=6)
            lbl.grid(row=r, column=c)
            cells[r][c] = lbl

    def set_all(color):
        for r in range(N):
            for c in range(N):
                cells[r][c].config(fg=color)

    state = {"i": 0}

    def step():
        if state["i"] >= len(schedule):
            root.destroy()
            return
        s = schedule[state["i"]]
        state["i"] += 1
        typ = s["type"]
        if typ == "cue":
            set_all(DIM_FG)
            r, c = char_to_rc(s["target_char"])
            cells[r][c].config(fg=CUE_FG)
            display = "␣" if s["target_char"] == "_" else s["target_char"]
            status.config(text=f"spell:  {display}   "
                               f"(letter {s['char_idx'] + 1})")
            root.after(timing["cue_ms"], lambda: (set_all(DIM_FG), step()))
        elif typ == "flash":
            for (r, c) in cells_for_code(s["code"]):
                cells[r][c].config(fg=ON_FG)
            mrec.push(s["code"], s["is_target"], s["char_idx"], s["rep"])
            root.after(timing["on_ms"], dim_then_gap)
        elif typ == "rest":
            set_all(DIM_FG)
            status.config(text="...")
            root.after(timing["rest_ms"], step)
        elif typ == "end":
            set_all(DIM_FG)
            status.config(text="DONE — stop EmotivPRO recording, then close this window")
            root.after(800, root.destroy)

    def dim_then_gap():
        set_all(DIM_FG)
        root.after(timing["off_ms"], step)

    started = {"go": False}

    def start(_event=None):
        if started["go"]:
            return
        started["go"] = True
        root.unbind("<Key>")
        root.unbind("<Button-1>")
        on_start(status, root, step)

    set_all(DIM_FG)
    status.config(
        text=f"START EmotivPRO recording, then click or press any key  "
             f"(Esc = exit full screen)"
    )
    root.bind("<Key>", start)
    root.bind("<Button-1>", start)
    root.mainloop()


def _run_countdown(status, root, step):
    """3-second on-screen countdown so EmotivPRO recording is definitely live."""
    for i in (3, 2, 1):
        status.config(text=str(i))
        root.update()
        time.sleep(1)
    status.config(text=f"")
    root.after(200, step)


# ---------------------------------------------------------------------------
# Headless driver
# ---------------------------------------------------------------------------
def run_headless(schedule, mrec, timing):
    print("[present] headless: starting in 3 s  (start EmotivPRO recording now)")
    for i in (3, 2, 1):
        print(f"          {i}...")
        time.sleep(1)
    for s in schedule:
        typ = s["type"]
        if typ == "cue":
            print(f"[present] cue '{s['target_char']}' "
                  f"(letter {s['char_idx'] + 1})")
            time.sleep(timing["cue_ms"] / 1000)
        elif typ == "flash":
            mrec.push(s["code"], s["is_target"], s["char_idx"], s["rep"])
            time.sleep((timing["on_ms"] + timing["off_ms"]) / 1000)
        elif typ == "rest":
            time.sleep(timing["rest_ms"] / 1000)
        elif typ == "end":
            print("[present] done — stop EmotivPRO recording now")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="6x6 P300 copy-spelling — Emotiv EPOC Flex (EmotivPRO workflow)"
    )
    ap.add_argument("--phrase", default="CSIRE",
                    help="target string to copy-spell")
    ap.add_argument("--reps", type=int, default=15,
                    help="stimulus repetitions per character")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window — dry-run for pipeline testing")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: recordings/session_<n>)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for flash order (default: random)")
    ap.add_argument("--on_ms",   type=int, default=100)
    ap.add_argument("--off_ms",  type=int, default=75)
    ap.add_argument("--cue_ms",  type=int, default=1000)
    ap.add_argument("--rest_ms", type=int, default=1500)
    args = ap.parse_args()

    phrase = args.phrase.upper().replace(" ", "_")
    for ch in phrase:
        char_to_rc(ch)
    timing = {
        "on_ms": args.on_ms, "off_ms": args.off_ms,
        "cue_ms": args.cue_ms, "rest_ms": args.rest_ms,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    rec_root = os.path.join(here, "recordings")
    os.makedirs(rec_root, exist_ok=True)
    if args.outdir:
        outdir = args.outdir
    else:
        n = 1
        while os.path.exists(os.path.join(rec_root, f"session_{n:03d}")):
            n += 1
        outdir = os.path.join(rec_root, f"session_{n:03d}")
    os.makedirs(outdir, exist_ok=True)

    # LSL marker outlet (optional — for any downstream LSL recorder)
    mrk_info = StreamInfo("P300_Markers", "Markers", 4, 0, "float32", "p300-mrk")
    mch = mrk_info.desc().append_child("channels")
    for name in ("code", "is_target", "char_idx", "rep"):
        mch.append_child("channel").append_child_value("label", name)
    marker_outlet = StreamOutlet(mrk_info)

    mrec = MarkerRecorder(outdir, marker_outlet)

    seed     = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rng      = random.Random(seed)
    schedule = build_schedule(phrase, args.reps, rng)
    n_flash  = sum(1 for s in schedule if s["type"] == "flash")
    secs     = n_flash * (args.on_ms + args.off_ms) / 1000
    print(f"[setup] phrase={phrase!r}  reps={args.reps}  seed={seed}  "
          f"SOA={args.on_ms + args.off_ms} ms  -> {outdir}")
    print(f"[setup] {len(phrase)} chars x {args.reps} reps = "
          f"{n_flash} flashes (~{secs:.0f}s)")

    if not args.headless:
        print("\n[action] START recording in EmotivPRO, then click/press any key in the window.")

    try:
        if args.headless:
            run_headless(schedule, mrec, timing)
        else:
            def on_start(status, root, step):
                _run_countdown(status, root, step)

            run_gui(schedule, mrec, timing, on_start)
    except KeyboardInterrupt:
        print("\n[present] interrupted by user")
    finally:
        mrec.flush()

    # Sanity check
    reconstructed = reconstruct_phrase(mrec.rows)
    match = "OK" if reconstructed == phrase else "MISMATCH"
    print(f"\n[result] target phrase   = {phrase!r}")
    print(f"[result] recorded phrase = {reconstructed!r}  [{match}]")
    print(f"[result] {len(mrec.rows)} marker flashes saved")

    # Session metadata
    meta = {
        "phrase": phrase,
        "reps": args.reps,
        "grid": GRID,
        "board": "emotiv_epoc_flex",
        "eeg_source": "EmotivPRO_manual_export",
        "timing_ms": timing,
        "soa_ms": args.on_ms + args.off_ms,
        "code_convention": "1-6=columns, 7-12=rows",
        "marker_channels": ["code", "is_target", "char_idx", "rep"],
        "wall_time_epoch": "unix_utc",
        "seed": seed,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] markers saved -> {outdir}")
    print(f"[done] STOP the EmotivPRO recording and export as EDF.")
    print(f"[done] Then run:  python decode_emotivpro.py {outdir} <path/to/export.edf>")


if __name__ == "__main__":
    main()
