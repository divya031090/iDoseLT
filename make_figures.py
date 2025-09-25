import os, yaml
from src.idoselt.plots import fig_cumulative_reward, fig_tacrolimus_patterns, bar_tradeoffs
from src.idoselt.eval import km_curves
import pandas as pd

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="data/artifacts", help="Path to artifacts")
    ap.add_argument("--synth", default="data/synth_cohort.parquet", help="Path to synthetic cohort")
    ap.add_argument("--out", default="figures", help="Figures dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fig2 = os.path.join(args.out, "fig2_km_survival.png")
    km_curves(pd.read_parquet(args.synth), pd.read_parquet(args.synth), fig2)

    fig3 = os.path.join(args.out, "fig3_cumulative_reward.png")
    fig_cumulative_reward(os.path.join(args.artifacts, "train_rewards.csv"), fig3)

    fig4 = os.path.join(args.out, "fig4_tacrolimus_patterns.png")
    fig_tacrolimus_patterns(args.synth, fig4)

    fig5 = os.path.join(args.out, "fig5_tradeoffs.png")
    bar_tradeoffs(fig5)

if __name__ == "__main__":
    main()
