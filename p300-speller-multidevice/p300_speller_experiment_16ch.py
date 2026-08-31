"""
P300 SPELLER EXPERIMENT (CUSTOM 16CH HEADSET) — 6x6 Farwell-Donchin
copy-spelling (OFFLINE calibration).

This is the *data-collection* counterpart to the offline tutorials (01-04).
Those decode a pre-recorded .mat; THIS script actually RUNS the paradigm:
it flashes a 6x6 grid, streams EEG + stimulus markers over LSL (exactly the
streaming layer the real system uses), and writes a fully-labelled recording
to disk that the offline decoder can train on.

Design (matches the email plan: "6x6 first, collect offline data while the
real-time pipeline is being validated"):

  * Paradigm : Farwell-Donchin 6x6 row/column speller.
               12 stimuli per repetition (codes 1-6 = COLUMNS, 7-12 = ROWS),
               each repetition flashes all 12 once in random order.
  * Mode     : COPY-SPELLING. The target string is known, so every flash is
               labelled target / non-target at record time -> supervised data.
  * Acquisition: the customized 16-channel headset, streamed over a serial
               (USB) link and parsed with EEG_Driver from
               eeg_serial_collect_16ch.py (see SerialAcquisition below). Everything
               downstream (LSL, markers, recorder) is unchanged.
  * Startup  : two explicit gates so nothing streams before you're ready:
                 1) connect to the headset over serial -> print status ->
                    press any key (in the CONSOLE) to open the GUI.
                 2) GUI opens dimmed  -> press any key (IN THE WINDOW) ->
                    LSL streaming (board + recorder) starts, then flashing
                    begins after a short settle.
  * Output   : recordings/<session>/  with
                   eeg.csv      lsl_timestamp + N EEG channels
                   markers.csv  lsl_timestamp, code, is_target, char_idx, rep
                   meta.json    grid, fs, channel names, timing, phrase
               -> feed to decode_recording.py for offline xDAWN+Riemannian.

Run (real experiment, GUI window):
    python p300_speller_experiment_16ch.py --phrase HELLO --reps 15

Run (headless pipeline dry-run, no window — validates streaming+recording):
    python p300_speller_experiment_16ch.py --phrase AB --reps 3 --headless

NOTE on timing: this uses Tkinter for zero-install runnability. For stricter
timing requirements you may want frame-locked presentation (PsychoPy); the
paradigm/scheduling here is renderer-agnostic so swapping the front-end is
localised to the presentation functions.
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
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

from eeg_serial_collect_16ch import EEG_Driver

# ---------------------------------------------------------------------------
# 0. The 6x6 speller grid (standard BCI Competition III layout; matches 03_*).
#    StimulusCode 1-6  -> columns 1-6
#    StimulusCode 7-12 -> rows    1-6
# ---------------------------------------------------------------------------
GRID = ["ABCDEF",
        "GHIJKL",
        "MNOPQR",
        "STUVWX",
        "YZ1234",
        "56789_"]
N = 6  # 6x6

N_EEG = 16  # fixed channel count of the customized headset


def char_to_rc(ch):
    """Return (row, col) 0-indexed of a character in the grid."""
    ch = ch.upper()
    for r, row in enumerate(GRID):
        c = row.find(ch)
        if c >= 0:
            return r, c
    raise ValueError(f"character {ch!r} is not on the 6x6 grid")


def char_to_target_codes(ch):
    """Return (col_code, row_code) -> the two stimulus codes that are TARGETS
    for this character. col_code in 1..6, row_code in 7..12."""
    r, c = char_to_rc(ch)
    return c + 1, r + 7


def code_pair_to_char(row_code, col_code):
    """row_code in 7..12, col_code in 1..6 -> grid letter (used by the decoder)."""
    return GRID[row_code - 7][col_code - 1]


def reconstruct_phrase(markers):
    """Sanity-check: rebuild the spelled phrase from recorded target flashes
    (lets us confirm the markers we saved actually match what we intended)."""
    per_char = {}
    for _t, code, is_target, char_idx, _rep in markers:
        if int(is_target) == 1:
            per_char.setdefault(int(char_idx), set()).add(int(code))
    phrase = []
    for ci in sorted(per_char):
        codes = per_char[ci]
        cols = [c for c in codes if 1 <= c <= 6]
        rows = [c for c in codes if 7 <= c <= 12]
        phrase.append(code_pair_to_char(rows[0], cols[0]) if cols and rows else "?")
    return "".join(phrase)


def cells_for_code(code):
    """Return the list of (row, col) grid cells that intensify for `code`."""
    if 1 <= code <= 6:                     # column
        c = code - 1
        return [(r, c) for r in range(N)]
    elif 7 <= code <= 12:                  # row
        r = code - 7
        return [(r, c) for c in range(N)]
    raise ValueError(code)


# ---------------------------------------------------------------------------
# 1. Build the flash schedule for the whole session.
#    Each repetition = all 12 codes once, in random order (no immediate repeat
#    across the rep boundary, standard practice to avoid adjacency artefacts).
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
            if order[0] == last_code and len(order) > 1:   # avoid back-to-back
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
# 2. Acquisition thread: customized 16ch serial headset -> LSL EEG outlet
#    (the producer). Frame parsing (0xAB ... 0xDC 0xBA, 16x 3-byte signed
#    channels) is delegated to EEG_Driver.read_sample() from
#    eeg_serial_collect_16ch.py -- this thread just loops it and forwards each
#    parsed sample to LSL, so the rest of the pipeline (Recorder, csv writing)
#    doesn't need to know the headset is serial-based.
# ---------------------------------------------------------------------------
class SerialAcquisition(threading.Thread):
    def __init__(self, eeg_driver, outlet, stop_event):
        super().__init__(daemon=True)
        self.eeg_driver = eeg_driver
        self.outlet, self.stop_event = outlet, stop_event

    def run(self):
        while not self.stop_event.is_set():
            channels = self.eeg_driver.read_sample()
            if channels is not None:
                self.outlet.push_sample(channels)


# ---------------------------------------------------------------------------
# 3. Recorder thread: subscribe to EEG + Markers inlets, log to memory, flush
#    to CSV at the end (the "hybrid" recorder from streaming lesson s5).
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
        # drain whatever is left in the buffers after the paradigm ends
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
            w.writerow(["lsl_timestamp", "code", "is_target", "char_idx", "rep"])
            for row in self.mrk_rows_out:
                t, code, tgt, ci, rep = row
                w.writerow([t, int(code), int(tgt), int(ci), int(rep)])
        print(f"[recorder] wrote {len(self.eeg_rows_out)} EEG rows -> {eeg_path}")
        print(f"[recorder] wrote {len(self.mrk_rows_out)} markers -> {mrk_path}")


def wait_for_keypress():
    """Block the CONSOLE (not a GUI) until any key is pressed."""
    try:
        import msvcrt
        msvcrt.getch()
    except ImportError:
        input()


# ---------------------------------------------------------------------------
# 4a. Presentation — Tkinter GUI (real experiment front-end).
# ---------------------------------------------------------------------------
DIM_FG = "#5a5a5a"
ON_FG = "#ffffff"
CUE_FG = "#34d058"
BG = "#000000"


def run_gui(schedule, marker_outlet, timing, phrase, on_start):
    """on_start() is called once, the moment the user presses a key/click on
    the ready screen -- this is where LSL streaming actually begins."""
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("P300 6x6 Speller — copy spelling")
    root.configure(bg=BG)
    root.geometry("680x760")
    root.attributes("-fullscreen", True)          # start full screen
    # Esc (or F11) toggles out of full screen so you are never trapped
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
    root.bind("<F11>",
              lambda e: root.attributes("-fullscreen",
                                        not root.attributes("-fullscreen")))

    big = tkfont.Font(family="Consolas", size=72, weight="bold")
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
            marker_outlet.push_sample([float(s["code"]), float(s["is_target"]),
                                       float(s["char_idx"]), float(s["rep"])])
            root.after(timing["on_ms"], dim_then_gap)
        elif typ == "rest":
            set_all(DIM_FG)
            status.config(text="...")
            root.after(timing["rest_ms"], step)
        elif typ == "end":
            status.config(text="DONE — you can close this window")
            root.after(800, root.destroy)

    def dim_then_gap():
        set_all(DIM_FG)
        root.after(timing["off_ms"], step)

    # --- start gate: pause on a ready screen until the user presses a key ----
    started = {"go": False}

    def start(_event=None):
        if started["go"]:
            return
        started["go"] = True
        root.unbind("<Key>")
        root.unbind("<Button-1>")
        status.config(text="starting stream...")
        root.update_idletasks()   # paint the status before the blocking call below
        on_start()                # <- LSL streaming (board + recorder) starts HERE
        status.config(text="")
        root.after(800, step)    # short settle, then first cue

    set_all(DIM_FG)
    status.config(text=f"Spell '{phrase}'  —  click mouse to start  "
                       f"(Esc to exit full screen)")
    root.bind("<Key>", start)
    root.bind("<Button-1>", start)
    root.mainloop()


# ---------------------------------------------------------------------------
# 4b. Presentation — headless driver (no window). Same schedule + timing, so
#     the streaming + recording pipeline is exercised end-to-end for a dry-run.
# ---------------------------------------------------------------------------
def run_headless(schedule, marker_outlet, timing):
    for s in schedule:
        typ = s["type"]
        if typ == "cue":
            print(f"[present] cue '{s['target_char']}' "
                  f"(letter {s['char_idx'] + 1})")
            time.sleep(timing["cue_ms"] / 1000)
        elif typ == "flash":
            marker_outlet.push_sample([float(s["code"]), float(s["is_target"]),
                                       float(s["char_idx"]), float(s["rep"])])
            time.sleep((timing["on_ms"] + timing["off_ms"]) / 1000)
        elif typ == "rest":
            time.sleep(timing["rest_ms"] / 1000)
        elif typ == "end":
            print("[present] done")


# ---------------------------------------------------------------------------
# 5. Main: wire acquisition + markers + recorder + presentation together.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="6x6 P300 copy-spelling experiment")
    ap.add_argument("--phrase", default="CSIRE",
                    help="target string to copy-spell (letters must be on the grid)")
    ap.add_argument("--reps", type=int, default=15,
                    help="stimulus repetitions per character (15 = dataset default)")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window — dry-run the streaming/recording pipeline")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: recordings/session_<n>)")
    ap.add_argument("--no-record", action="store_true",
                    help="don't record locally — just present + broadcast LSL "
                         "(use when a 2nd computer runs record_session.py)")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for flash order")
    ap.add_argument("--serial-port", default="COM3",
                    help="COM port the customized 16ch headset is attached to")
    ap.add_argument("--baud", type=int, default=3000000, help="serial baud rate")
    # timing (ms) — defaults give 0.175 s SOA == the 03_* tutorial assumption
    ap.add_argument("--on_ms", type=int, default=100)
    ap.add_argument("--off_ms", type=int, default=75)
    ap.add_argument("--cue_ms", type=int, default=1000)
    ap.add_argument("--rest_ms", type=int, default=1500)
    args = ap.parse_args()

    phrase = args.phrase.upper()
    for ch in phrase:
        char_to_rc(ch)  # validate up-front
    timing = {"on_ms": args.on_ms, "off_ms": args.off_ms,
              "cue_ms": args.cue_ms, "rest_ms": args.rest_ms}

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

    # --- gate 1: connect to the headset, show status, pause for a console key --
    n_eeg = N_EEG
    eeg_names = [f"ch{i}" for i in range(n_eeg)]
    print(f"[headset] connecting on {args.serial_port} @ {args.baud} baud ...")
    eeg_driver = EEG_Driver(serial_port=args.serial_port, baud_rate=args.baud)
    print(f"[headset] connected  ({n_eeg}ch serial)")

    stop_event = threading.Event()
    ready_event = threading.Event()
    state = SimpleNamespace(acq=None, rec=None)

    # everything from here on can be interrupted (Ctrl+C, closed window, ...);
    # this try/finally guarantees the serial port is always released.
    try:
        if args.headless:
            print("[headset] --headless: skipping the console/GUI key-press gates")
        else:
            print("\npress any key to open the speller GUI...")
            wait_for_keypress()

        # --- LSL outlets (registration only; no samples flow until gate 2) ---
        # nominal rate 0 == irregular: the serial link doesn't tick at a fixed
        # rate, so each pushed sample is timestamped as it arrives (local_clock()).
        eeg_info = StreamInfo("P300_EEG", "EEG", n_eeg, 0, "float32", "p300-eeg")
        chns = eeg_info.desc().append_child("channels")
        for name in eeg_names:
            chns.append_child("channel").append_child_value("label", name)
        eeg_outlet = StreamOutlet(eeg_info)

        mrk_info = StreamInfo("P300_Markers", "Markers", 4, 0, "float32", "p300-mrk")
        mch = mrk_info.desc().append_child("channels")
        for name in ("code", "is_target", "char_idx", "rep"):
            mch.append_child("channel").append_child_value("label", name)
        marker_outlet = StreamOutlet(mrk_info)

        print(f"[setup] phrase={phrase!r}  reps={args.reps}  "
              f"SOA={args.on_ms + args.off_ms} ms  -> {outdir}")

        # --- gate 2: streaming only starts once the GUI (or headless) fires --
        def start_streaming():
            state.acq = SerialAcquisition(eeg_driver, eeg_outlet, stop_event)
            state.acq.start()
            if args.no_record:
                print("[setup] --no-record: presenting + broadcasting LSL only "
                      "(record on the 2nd computer with record_session.py)")
            else:
                state.rec = Recorder(n_eeg, eeg_names, outdir, stop_event, ready_event)
                state.rec.start()
                ready_event.wait(timeout=12)     # wait until inlets are open

        # --- schedule ----------------------------------------------------------
        rng = random.Random(args.seed)
        schedule = build_schedule(phrase, args.reps, rng)
        n_flash = sum(1 for s in schedule if s["type"] == "flash")
        secs = n_flash * (args.on_ms + args.off_ms) / 1000
        print(f"[present] {len(phrase)} chars x {args.reps} reps = {n_flash} flashes "
              f"(~{secs:.0f}s of flashing)")
        if args.headless:
            start_streaming()                    # no GUI gate -- start right away
            run_headless(schedule, marker_outlet, timing)
        else:
            run_gui(schedule, marker_outlet, timing, phrase, start_streaming)
    except KeyboardInterrupt:
        print("\n[present] interrupted by user")
    finally:
        time.sleep(0.8)                          # let last epochs stream in
        stop_event.set()
        if state.rec is not None:
            state.rec.join(timeout=5)
        if state.acq is not None:
            state.acq.join(timeout=5)
        eeg_driver.ser.close()

    # --- result summary --------------------------------------------------------
    rec = state.rec
    fs_effective = None
    if rec is not None and len(rec.eeg_rows_out) > 1:
        t_first, t_last = rec.eeg_rows_out[0][0], rec.eeg_rows_out[-1][0]
        n_samples = len(rec.eeg_rows_out)
        if t_last > t_first:
            fs_effective = (n_samples - 1) / (t_last - t_first)
    if rec is not None:
        reconstructed = reconstruct_phrase(rec.mrk_rows_out)
        match = "OK" if reconstructed == phrase else "MISMATCH"
        print(f"\n[result] target phrase   = {phrase!r}")
        print(f"[result] recorded phrase = {reconstructed!r}  [{match}]")
        print(f"[result] {len(rec.eeg_rows_out)} EEG samples, "
              f"{len(rec.mrk_rows_out)} marker flashes recorded")
        print(f"[result] eeg.csv     -> {os.path.join(outdir, 'eeg.csv')}")
        print(f"[result] markers.csv -> {os.path.join(outdir, 'markers.csv')}")
    else:
        print("\n[result] streaming never started (closed before the GUI key-press)"
              if not args.headless and not args.no_record
              else "\n[result] --no-record: nothing saved locally by this process")

    # --- session metadata ----------------------------------------------------
    meta = {
        "phrase": phrase,
        "reps": args.reps,
        "grid": GRID,
        "fs": fs_effective,   # measured from recorded LSL timestamps (irregular-rate stream)
        "n_eeg": n_eeg,
        "eeg_names": eeg_names,
        "board": "custom_16ch_serial",
        "serial_port": args.serial_port,
        "baud_rate": args.baud,
        "timing_ms": timing,
        "soa_ms": args.on_ms + args.off_ms,
        "code_convention": "1-6=columns, 7-12=rows",
        "marker_channels": ["code", "is_target", "char_idx", "rep"],
        "seed": args.seed,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] session saved -> {outdir}")
    print(f"[done] decode it with:  python decode_recording.py {outdir}")


if __name__ == "__main__":
    main()
