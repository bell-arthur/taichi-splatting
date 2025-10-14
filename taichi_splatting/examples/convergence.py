import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def ema(x, alpha):
    """Exponential moving average smoothing."""
    if alpha <= 0:
        return x.values
    out = np.empty_like(x.values, dtype=float)
    out[0] = x.values[0]
    for i in range(1, len(x)):
        out[i] = alpha * x.values[i] + (1 - alpha) * out[i-1]
    return out

def find_convergence(df, method="threshold", tol=0.1, rel=0.98,
                     slope_window=10, slope_delta=0.02, smooth_alpha=0.0,
                     pick="first"):
    s = df["psnr"].copy()
    s = pd.Series(ema(s, smooth_alpha), index=s.index)

    best_idx = int(s.idxmax())
    best_psnr = float(s.loc[best_idx])
    best_iter = int(df.loc[best_idx, "iteration"])
    best_time = float(df.loc[best_idx, "time"])

    if method == "threshold":
        target = best_psnr - tol
        hits = s.index[(s >= target)]
    elif method == "relative":
        start_psnr = float(s.iloc[0])
        target = start_psnr + rel * (best_psnr - start_psnr)
        hits = s.index[(s >= target)]
    elif method == "slope":
        diffs = s.diff().abs()
        roll = diffs.rolling(slope_window, min_periods=slope_window).mean()
        plateau_mask = (roll < slope_delta)
        # rising edges into low-slope region (start of plateau)
        transitions = plateau_mask & ~plateau_mask.shift(1, fill_value=False)
        hits = transitions.index[transitions]
    else:
        raise ValueError("Unknown method")

    if len(hits) == 0:
        return dict(
            best_psnr=best_psnr,
            best_iter=best_iter,
            best_time=best_time,
            converged=False,
            convergence_iter=None,
            convergence_time=None,
            convergence_psnr=None,
        )

    conv_idx = int(hits[0] if pick == "first" else hits[-1])
    return dict(
        best_psnr=best_psnr,
        best_iter=best_iter,
        best_time=best_time,
        converged=True,
        convergence_iter=int(df.loc[conv_idx, "iteration"]),
        convergence_time=float(df.loc[conv_idx, "time"]),
        convergence_psnr=float(df.loc[conv_idx, "psnr"]),
    )

def process_file(file, args):
    df = pd.read_csv(file)
    if not {"iteration", "time", "psnr"}.issubset(df.columns):
        raise ValueError(f"{file} missing required columns.")
    result = find_convergence(
        df, method=args.method, tol=args.tol, rel=args.rel,
        slope_window=args.slope_window, slope_delta=args.slope_delta,
        smooth_alpha=args.ema, pick=args.pick,
    )
    result["filename"] = file.name
    return result

def main():
    ap = argparse.ArgumentParser(description="Compute convergence metrics for one or many CSVs.")
    ap.add_argument("input", type=Path, help="CSV file or directory containing CSVs.")
    ap.add_argument("--out", type=Path, default=Path("convergence_summary.csv"),
                    help="Output CSV file for summary results.")
    ap.add_argument("--method", choices=["threshold","relative","slope"], default="threshold")
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--rel", type=float, default=0.98)
    ap.add_argument("--slope-window", type=int, default=10)
    ap.add_argument("--slope-delta", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--pick", choices=["first","last"], default="first",
                    help="Select first or last detected plateau/crossing")
    args = ap.parse_args()

    def fmt2(x):
        return f"{x:.2f}" if x is not None and not pd.isna(x) else "NA"

    if args.input.is_dir():
        files = sorted(args.input.glob("*.csv"))
    elif args.input.is_file():
        files = [args.input]
    else:
        raise SystemExit(f"Invalid input path: {args.input}")

    results = []
    for f in files:
        try:
            res = process_file(f, args)
            results.append(res)
            print(f"[OK] {f.name}: converged={res['converged']} "
                  f"at iter={res['convergence_iter']} (PSNR={fmt2(res['convergence_psnr'])})")
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    if results:
        df = pd.DataFrame(results)
        cols = [
            "filename",
            "convergence_iter",
            "convergence_time",
            "convergence_psnr",
            "best_iter",
            "best_time",
            "best_psnr",
        ]
        df = df.reindex(columns=cols)
        df.to_csv(args.out, index=False, float_format="%.2f")
        print(f"\nSaved summary to {args.out.resolve()}")
    else:
        print("No valid CSVs processed.")

if __name__ == "__main__":
    main()