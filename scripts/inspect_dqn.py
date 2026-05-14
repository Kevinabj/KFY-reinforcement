"""Quick learning-curve inspector for any logged run."""

import argparse
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    df["window20"] = df["episode_return"].rolling(20).mean()
    print(f"file:         {args.csv}")
    print(f"episodes:     {len(df)}")
    print(f"total_steps:  {int(df['step'].iloc[-1])}")
    print(f"max_return:   {df['episode_return'].max():.1f}")
    print(f"final 20-ep:  {df['window20'].iloc[-1]:.1f}")
    print(f"final 100-ep: {df['episode_return'].tail(100).mean():.1f} +/- {df['episode_return'].tail(100).std():.1f}")
    print(f"\n20-ep rolling mean across training:")
    for s in [5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 200_000, 300_000]:
        sub = df[df.step <= s]
        if len(sub) == 0:
            continue
        ep = int(sub["episode"].iloc[-1])
        win = sub["window20"].iloc[-1]
        print(f"  step {s:>6d}  ep {ep:>4d}  window20 = {win:.1f}")


if __name__ == "__main__":
    main()
