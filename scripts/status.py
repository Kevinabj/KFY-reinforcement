r"""Quick training status across all CSVs in logs/.

Shows for every (algo, env, seed) found:
  - last step / total steps from run_all.py MATRIX
  - last episode return
  - rolling-20 episode return
  - ETA for in-progress runs (based on recent throughput)
  - last modified time
  - whether the run looks complete (last step >= target * 0.99)

Run from applied_project/ with:
    .\.venv\Scripts\python.exe scripts\status.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the matrix so we know the target step count per (algo, env).
from scripts.run_all import MATRIX  # noqa: E402
from scripts.run_ablations import SWEEPS  # noqa: E402

TARGET = {(algo, env): total for algo, env, total, _, _ in MATRIX}

ABL_TARGET = {sweep: spec["total_steps"] for sweep, spec in SWEEPS.items()}


def fmt_time(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    log_dir = ROOT / "logs"
    csvs = sorted(
        log_dir.glob("*/*/seed*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    csvs = [p for p in csvs if "random" not in p.parts]
    if not csvs:
        print("No CSVs in logs/. Has anything trained yet?")
        return

    print(f"{'algo':<5} {'env':<28} {'seed':<5} "
          f"{'step / target':<24} {'last ret':>9} {'roll20':>9} "
          f"{'ETA':>9} {'last modified':<20} status")
    print("-" * 130)
    total_eta_seconds = 0.0
    for p in csvs:
        algo, env_id, fname = p.relative_to(log_dir).parts
        seed = int(fname.replace("seed", "").replace(".csv", ""))
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            print(f"{algo:<5} {env_id:<28} {seed:<5} (empty CSV)")
            continue
        if df.empty:
            continue
        target = TARGET.get((algo, env_id), None)
        last_step = int(df["step"].iloc[-1])
        last_ret = float(df["episode_return"].iloc[-1])
        roll20 = float(df["episode_return"].tail(20).mean())
        mtime = p.stat().st_mtime
        if target is None:
            sp = f"{last_step:>10,}"
            status = "?"
            pct = None
        else:
            sp = f"{last_step:>10,} / {target:>10,}"
            pct = last_step / target
            status = "DONE" if pct >= 0.99 else f"{pct*100:.0f}%"

        # ETA based on the last 20 episodes' env-steps-per-second rate
        eta_s = None
        if target is not None and pct is not None and pct < 0.99 and len(df) >= 5:
            tail = df.tail(min(20, len(df)))
            step_delta = float(tail["step"].iloc[-1]) - float(tail["step"].iloc[0])
            wall_delta = float(tail["wall_clock_s"].iloc[-1]) - float(tail["wall_clock_s"].iloc[0])
            if wall_delta > 0:
                rate = step_delta / wall_delta  # env steps / sec
                remaining = target - last_step
                eta_s = remaining / max(rate, 1e-6)
                total_eta_seconds += eta_s

        if eta_s is None:
            eta_str = "--"
        else:
            mins = int(eta_s // 60)
            secs = int(eta_s % 60)
            eta_str = f"{mins:>3d}m{secs:02d}s"

        active = (datetime.now().timestamp() - mtime) < 60
        marker = "<- ACTIVE" if active else ""
        print(f"{algo:<5} {env_id:<28} {seed:<5} {sp:<24} "
              f"{last_ret:>9.1f} {roll20:>9.1f} {eta_str:>9} "
              f"{fmt_time(mtime):<20} {status} {marker}")

    # Account for runs in run_all.py MATRIX that have not even started yet.
    missing_total_steps = 0
    seen = {(p.parts[-3], p.parts[-2]) for p in csvs}
    # We use seed-0 CSV existence as proxy; for per-seed status, see above.
    for algo, env, total, _, _ in MATRIX:
        # Count seeds 0, 1, 2 not yet on disk for any seed.
        for s in (0, 1, 2):
            cand = log_dir / algo / env / f"seed{s}.csv"
            if not cand.exists():
                missing_total_steps += total

    # If any active run, also estimate time for the queued missing runs using
    # that active run's rate.
    if missing_total_steps > 0 and total_eta_seconds > 0:
        # crude: assume similar per-step cost on average as the current active run
        active_rows = [p for p in csvs if (datetime.now().timestamp() - p.stat().st_mtime) < 60]
        if active_rows:
            ap = active_rows[0]
            adf = pd.read_csv(ap)
            tail = adf.tail(min(20, len(adf)))
            step_delta = float(tail["step"].iloc[-1]) - float(tail["step"].iloc[0])
            wall_delta = float(tail["wall_clock_s"].iloc[-1]) - float(tail["wall_clock_s"].iloc[0])
            rate = step_delta / wall_delta if wall_delta > 0 else 80.0
            queued = missing_total_steps / max(rate, 1e-6)
            print(f"\nEstimated time for queued (unstarted) runs at current rate: "
                  f"{int(queued // 60)}m{int(queued % 60):02d}s")

    if total_eta_seconds > 0:
        h = int(total_eta_seconds // 3600)
        m = int((total_eta_seconds % 3600) // 60)
        s = int(total_eta_seconds % 60)
        print(f"Estimated time to finish in-progress runs:           {h}h{m:02d}m{s:02d}s")

    # ----------------------------------------------------------------- ablations
    abl_dir = ROOT / "logs_ablation"
    # Completed runs land at logs_ablation/<sweep>/<value>/seed<N>.csv.
    # In-progress runs are still being written under the _tmp subtree:
    #   logs_ablation/<sweep>/<value>/_tmp/<algo>/<env>/seed<N>.csv
    abl_done = list(abl_dir.glob("*/*/seed*.csv")) if abl_dir.exists() else []
    abl_inprogress = list(abl_dir.glob("*/*/_tmp/*/*/seed*.csv")) if abl_dir.exists() else []

    # Build a unified list of (sweep, value, seed, csv_path), preferring done over in-progress.
    seen_keys: set = set()
    unified: list = []
    for p in sorted(abl_done, key=lambda x: x.stat().st_mtime, reverse=True):
        sweep_name = p.parts[-3]
        value_label = p.parts[-2]
        seed = int(p.stem.replace("seed", ""))
        key = (sweep_name, value_label, seed)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unified.append((sweep_name, value_label, seed, p))
    for p in sorted(abl_inprogress, key=lambda x: x.stat().st_mtime, reverse=True):
        sweep_name = p.parts[-6]
        value_label = p.parts[-5]
        seed = int(p.stem.replace("seed", ""))
        key = (sweep_name, value_label, seed)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unified.append((sweep_name, value_label, seed, p))
    abl_csvs = unified

    if abl_csvs:
        print("\n" + "=" * 80)
        print("ABLATIONS (logs_ablation/)")
        print(f"{'sweep':<25} {'value':<10} {'seed':<5} {'step / target':<24} "
              f"{'last ret':>9} {'roll20':>9} {'ETA':>9} {'last modified':<20} status")
        print("-" * 130)
        total_abl_eta = 0.0
        for sweep_name, value_label, seed, p in abl_csvs:
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                print(f"{sweep_name:<25} {value_label:<10} {seed:<5} (empty CSV)")
                continue
            if df.empty:
                continue
            target = ABL_TARGET.get(sweep_name)
            last_step = int(df["step"].iloc[-1])
            last_ret = float(df["episode_return"].iloc[-1])
            roll20 = float(df["episode_return"].tail(20).mean())
            mtime = p.stat().st_mtime
            if target is None:
                sp = f"{last_step:>10,}"
                status = "?"
                pct = None
            else:
                sp = f"{last_step:>10,} / {target:>10,}"
                pct = last_step / target
                status = "DONE" if pct >= 0.99 else f"{pct*100:.0f}%"

            eta_s = None
            if target is not None and pct is not None and pct < 0.99 and len(df) >= 5:
                tail = df.tail(min(20, len(df)))
                step_delta = float(tail["step"].iloc[-1]) - float(tail["step"].iloc[0])
                wall_delta = float(tail["wall_clock_s"].iloc[-1]) - float(tail["wall_clock_s"].iloc[0])
                if wall_delta > 0:
                    rate = step_delta / wall_delta
                    eta_s = (target - last_step) / max(rate, 1e-6)
                    total_abl_eta += eta_s

            if eta_s is None:
                eta_str = "--"
            else:
                mins = int(eta_s // 60)
                secs = int(eta_s % 60)
                eta_str = f"{mins:>3d}m{secs:02d}s"

            active = (datetime.now().timestamp() - mtime) < 60
            marker = "<- ACTIVE" if active else ""
            print(f"{sweep_name:<25} {value_label:<10} {seed:<5} {sp:<24} "
                  f"{last_ret:>9.1f} {roll20:>9.1f} {eta_str:>9} "
                  f"{fmt_time(mtime):<20} {status} {marker}")

        # Also estimate time for queued ablation runs (in SWEEPS but no CSV yet,
        # neither completed nor in-progress).
        missing_abl_steps = 0
        for sweep_name, spec in SWEEPS.items():
            for value_label, _ in spec["values"]:
                for s in (0, 1, 2):
                    if (sweep_name, value_label, s) not in seen_keys:
                        missing_abl_steps += spec["total_steps"]

        # Use rate from the most recently-active ablation row if any.
        if missing_abl_steps > 0:
            active_paths = [
                p for (_, _, _, p) in abl_csvs
                if (datetime.now().timestamp() - p.stat().st_mtime) < 60
            ]
            if active_paths:
                ap = active_paths[0]
                adf = pd.read_csv(ap)
                tail = adf.tail(min(20, len(adf)))
                step_delta = float(tail["step"].iloc[-1]) - float(tail["step"].iloc[0])
                wall_delta = float(tail["wall_clock_s"].iloc[-1]) - float(tail["wall_clock_s"].iloc[0])
                rate = step_delta / wall_delta if wall_delta > 0 else 60.0
                queued_s = missing_abl_steps / max(rate, 1e-6)
                qh = int(queued_s // 3600)
                qm = int((queued_s % 3600) // 60)
                qs = int(queued_s % 60)
                print(f"Estimated time for queued (unstarted) ablations:     "
                      f"{qh}h{qm:02d}m{qs:02d}s")

        if total_abl_eta > 0:
            h = int(total_abl_eta // 3600)
            m = int((total_abl_eta % 3600) // 60)
            s = int(total_abl_eta % 60)
            print(f"Estimated time to finish in-progress ablations:      {h}h{m:02d}m{s:02d}s")
    else:
        print("\nNo ablation CSVs yet (logs_ablation/ is empty or missing).")


if __name__ == "__main__":
    main()
