"""
P300 SPELLER EXPERIMENT — Emotiv EPOC Flex, 6x6 Farwell-Donchin
copy-spelling (OFFLINE calibration) — CORTEX API VERSION.

Uses the Emotiv Cortex WebSocket API to stream raw EEG directly into Python
(via emotiv_cortex_acquisition.py) AND records stimulus markers, all saved
to CSV locally. No manual EmotivPRO recording needed.

Prerequisites:
    pip install websocket-client python-dotenv pylsl numpy
    Emotiv Cortex app (or EmotivPRO) must be running before launch.
    Cortex API EEG license required (error -32232 means license is missing).
    Developer credentials in .env:
        EMOTIV_CLIENT_ID=...
        EMOTIV_CLIENT_SECRET=...
        EMOTIV_HEADSET_ID=...   (optional; default = first headset Cortex finds)

Run (real experiment):
    python p300_speller_experiment_emotiv_cortex.py --phrase HELLO --reps 15

Run (headless pipeline dry-run, no window):
    python p300_speller_experiment_emotiv_cortex.py --phrase AB --reps 3 --headless

Override credentials on the command line:
    python p300_speller_experiment_emotiv_cortex.py --client-id X --client-secret Y ...

Output (in recordings/session_<n>/):
    eeg.csv      lsl_timestamp + EEG channel columns
    markers.csv  lsl_timestamp, wall_time, code, is_target, char_idx, rep
    meta.json    grid, fs, channel names, timing, phrase
"""
import argparse
import csv
import json
import os
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

from emotiv_cortex_acquisition import CortexAcquisition

load_dotenv()

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
# Recorder thread — LSL EEG + Markers inlets -> CSV
# ---------------------------------------------------------------------------
class Recorder(threading.Thread):
    def __init__(self, n_eeg, eeg_names, outdir, stop_event, ready_event):
        super().__init__(daemon=True)
        self.n_eeg, self.eeg_names, self.outdir = n_eeg, eeg_names, outdir
        self.stop_event, self.ready_event = stop_event, ready_event
        self.eeg_rows_out, self.mrk_rows_out = [], []

    def run(self):
        eeg_streams = resolve_byprop("name", "P300_EEG", timeout=10)
        mrk_streams = resolve_byprop("name", "P300_Markers", timeout=10)
        if not eeg_streams or not mrk_streams:
            print("[recorder] could not resolve LSL streams; aborting.")
            self.ready_event.set()
            return
        eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=600)
        mrk_inlet = StreamInlet(mrk_streams[0], max_buflen=600)
        self.ready_event.set()
        while not self.stop_event.is_set():
            s, ts = eeg_inlet.pull_chunk(timeout=0.0)
            if ts:
                for row, t in zip(s, ts):
                    self.eeg_rows_out.append([t] + list(row))
            s, ts = mrk_inlet.pull_chunk(timeout=0.0)
            if ts:
                for row, t in zip(s, ts):
                    self.mrk_rows_out.append([t] + list(row))
            time.sleep(0.005)
        for inlet, sink in ((eeg_inlet, self.eeg_rows_out),
                            (mrk_inlet, self.mrk_rows_out)):
            s, ts = inlet.pull_chunk(timeout=0.2)
            if ts:
                for row, t in zip(s, ts):
                    sink.append([t] + list(row))
        self._flush()

    def _flush(self):
        eeg_path = os.path.join(self.outdir, "eeg.csv")
        mrk_path = os.path.join(self.outdir, "markers.csv")
        with open(eeg_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lsl_timestamp"] + list(self.eeg_names))
            w.writerows(self.eeg_rows_out)
        with open(mrk_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lsl_timestamp", "wall_time", "code",
                        "is_target", "char_idx", "rep"])
            for row in self.mrk_rows_out:
                t, code, wt, tgt, ci, rep = row
                w.writerow([t, f"{wt:.6f}", int(code), int(tgt), int(ci), int(rep)])
        print(f"[recorder] wrote {len(self.eeg_rows_out)} EEG rows  -> {eeg_path}")
        print(f"[recorder] wrote {len(self.mrk_rows_out)} markers   -> {mrk_path}")


def wait_for_keypress():
    try:
        import msvcrt
        msvcrt.getch()
    except ImportError:
        input()


# ---------------------------------------------------------------------------
# GUI presentation (Tkinter)
# ---------------------------------------------------------------------------
DIM_FG = "#5a5a5a"
ON_FG  = "#ffffff"
CUE_FG = "#34d058"
BG     = "#000000"


def run_gui(schedule, marker_outlet, timing, phrase, on_start):
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("P300 6x6 Speller — Emotiv EPOC Flex (Cortex)")
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
            status.config(text=f"spell:  {s['target_char']}   "
                               f"(letter {s['char_idx'] + 1})")
            root.after(timing["cue_ms"], lambda: (set_all(DIM_FG), step()))
        elif typ == "flash":
            for (r, c) in cells_for_code(s["code"]):
                cells[r][c].config(fg=ON_FG)
            marker_outlet.push_sample([
                float(s["code"]), float(time.time()),
                float(s["is_target"]), float(s["char_idx"]), float(s["rep"]),
            ])
            root.after(timing["on_ms"], dim_then_gap)
        elif typ == "rest":
            set_all(DIM_FG)
            status.config(text="...")
            root.after(timing["rest_ms"], step)
        elif typ == "end":
            set_all(DIM_FG)
            status.config(text="DONE — you can close this window")
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
        status.config(text="starting stream...")
        root.update_idletasks()
        on_start()
        status.config(text="")
        root.after(800, step)

    set_all(DIM_FG)
    status.config(text=f"Spell '{phrase}'  —  click or press any key to start  "
                       f"(Esc = exit full screen)")
    root.bind("<Key>", start)
    root.bind("<Button-1>", start)
    root.mainloop()


# ---------------------------------------------------------------------------
# Headless driver
# ---------------------------------------------------------------------------
def run_headless(schedule, marker_outlet, timing):
    for s in schedule:
        typ = s["type"]
        if typ == "cue":
            print(f"[present] cue '{s['target_char']}' "
                  f"(letter {s['char_idx'] + 1})")
            time.sleep(timing["cue_ms"] / 1000)
        elif typ == "flash":
            marker_outlet.push_sample([
                float(s["code"]), float(time.time()),
                float(s["is_target"]), float(s["char_idx"]), float(s["rep"]),
            ])
            time.sleep((timing["on_ms"] + timing["off_ms"]) / 1000)
        elif typ == "rest":
            time.sleep(timing["rest_ms"] / 1000)
        elif typ == "end":
            print("[present] done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="6x6 P300 copy-spelling — Emotiv EPOC Flex via Cortex API"
    )
    ap.add_argument("--phrase", default="CSIRE",
                    help="target string to copy-spell")
    ap.add_argument("--reps", type=int, default=15,
                    help="stimulus repetitions per character")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI — dry-run the streaming/recording pipeline")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: recordings/session_<n>)")
    ap.add_argument("--no-record", action="store_true",
                    help="present + broadcast LSL only (no local CSV)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--client-id", default=None,
                    help="Emotiv Client ID (default: EMOTIV_CLIENT_ID from .env)")
    ap.add_argument("--client-secret", default=None,
                    help="Emotiv Client Secret (default: EMOTIV_CLIENT_SECRET from .env)")
    ap.add_argument("--headset-id", default=None,
                    help="Emotiv headset ID (default: EMOTIV_HEADSET_ID from "
                         ".env, else first found)")
    ap.add_argument("--on_ms",   type=int, default=100)
    ap.add_argument("--off_ms",  type=int, default=75)
    ap.add_argument("--cue_ms",  type=int, default=1000)
    ap.add_argument("--rest_ms", type=int, default=1500)
    args = ap.parse_args()

    client_id     = args.client_id     or os.environ.get("EMOTIV_CLIENT_ID")
    client_secret = args.client_secret or os.environ.get("EMOTIV_CLIENT_SECRET")
    headset_id    = args.headset_id    or os.environ.get("EMOTIV_HEADSET_ID")
    if not client_id or not client_secret:
        ap.error(
            "Cortex credentials not found. "
            "Set EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET in .env, "
            "or pass --client-id / --client-secret."
        )

    phrase = args.phrase.upper()
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

    stop_event  = threading.Event()
    ready_event = threading.Event()
    state       = SimpleNamespace(acq=None, rec=None)

    # --- Gate 1: connect to Cortex + authenticate + discover channels -------
    print("[cortex] connecting ...")
    acq = CortexAcquisition(client_id, client_secret, stop_event,
                            headset_id=headset_id)
    acq.connect()

    n_eeg     = len(acq.channel_names)
    eeg_names = acq.channel_names
    print(f"[headset] EPOC Flex ready  ({n_eeg} channels)")

    try:
        if not args.headless:
            print("\nPress any key to open the speller GUI...")
            wait_for_keypress()

        # LSL outlets — channel count known after connect()
        eeg_info = StreamInfo("P300_EEG", "EEG", n_eeg, 0, "float32", "p300-eeg-emotiv")
        chns = eeg_info.desc().append_child("channels")
        for name in eeg_names:
            chns.append_child("channel").append_child_value("label", name)
        eeg_outlet = StreamOutlet(eeg_info)
        acq.set_outlet(eeg_outlet)

        # Marker stream: code, wall_time, is_target, char_idx, rep
        mrk_info = StreamInfo("P300_Markers", "Markers", 5, 0, "float32", "p300-mrk")
        mch = mrk_info.desc().append_child("channels")
        for name in ("code", "wall_time", "is_target", "char_idx", "rep"):
            mch.append_child("channel").append_child_value("label", name)
        marker_outlet = StreamOutlet(mrk_info)

        print(f"[setup] phrase={phrase!r}  reps={args.reps}  "
              f"SOA={args.on_ms + args.off_ms} ms  -> {outdir}")

        # --- Gate 2: streaming starts on GUI trigger ------------------------
        def start_streaming():
            state.acq = acq
            acq.start()
            if args.no_record:
                print("[setup] --no-record: broadcasting LSL only")
            else:
                state.rec = Recorder(n_eeg, eeg_names, outdir,
                                     stop_event, ready_event)
                state.rec.start()
                ready_event.wait(timeout=12)

        rng      = random.Random(args.seed)
        schedule = build_schedule(phrase, args.reps, rng)
        n_flash  = sum(1 for s in schedule if s["type"] == "flash")
        secs     = n_flash * (args.on_ms + args.off_ms) / 1000
        print(f"[present] {len(phrase)} chars x {args.reps} reps = "
              f"{n_flash} flashes (~{secs:.0f}s)")

        if args.headless:
            start_streaming()
            run_headless(schedule, marker_outlet, timing)
        else:
            run_gui(schedule, marker_outlet, timing, phrase, start_streaming)

    except KeyboardInterrupt:
        print("\n[present] interrupted by user")
    finally:
        time.sleep(0.8)
        stop_event.set()
        if state.rec is not None:
            state.rec.join(timeout=5)
        if state.acq is not None:
            state.acq.join(timeout=5)

    # --- Result summary ------------------------------------------------------
    rec = state.rec
    fs_effective = None
    if rec is not None and len(rec.eeg_rows_out) > 1:
        t0, t1 = rec.eeg_rows_out[0][0], rec.eeg_rows_out[-1][0]
        if t1 > t0:
            fs_effective = (len(rec.eeg_rows_out) - 1) / (t1 - t0)

    if rec is not None:
        reconstructed = reconstruct_phrase(rec.mrk_rows_out)
        match = "OK" if reconstructed == phrase else "MISMATCH"
        print(f"\n[result] target phrase   = {phrase!r}")
        print(f"[result] recorded phrase = {reconstructed!r}  [{match}]")
        print(f"[result] {len(rec.eeg_rows_out)} EEG samples, "
              f"{len(rec.mrk_rows_out)} marker flashes recorded")
        if fs_effective:
            print(f"[result] effective sample rate: {fs_effective:.1f} Hz")
        print(f"[result] eeg.csv     -> {os.path.join(outdir, 'eeg.csv')}")
        print(f"[result] markers.csv -> {os.path.join(outdir, 'markers.csv')}")
    else:
        print("\n[result] nothing recorded")

    # --- Session metadata ----------------------------------------------------
    meta = {
        "phrase": phrase,
        "reps": args.reps,
        "grid": GRID,
        "fs": fs_effective,
        "n_eeg": n_eeg,
        "eeg_names": eeg_names,
        "board": "emotiv_epoc_flex",
        "headset_id": acq.headset_id,
        "eeg_source": "cortex_api",
        "timing_ms": timing,
        "soa_ms": args.on_ms + args.off_ms,
        "code_convention": "1-6=columns, 7-12=rows",
        "marker_channels": ["code", "wall_time", "is_target", "char_idx", "rep"],
        "seed": args.seed,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] session saved -> {outdir}")
    print(f"[done] decode with:  python decode_recording.py {outdir}")


if __name__ == "__main__":
    main()
