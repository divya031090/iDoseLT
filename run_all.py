import os, yaml
from src.idoselt.train import train
from src.idoselt.plots import fig_cumulative_reward, fig_tacrolimus_patterns, bar_tradeoffs
from src.idoselt.eval import km_curves
import pandas as pd

def main():
    art = train("configs/config.yaml")
    out_dir = art["out_dir"]
    figures_dir = art["figures_dir"]
    synth = art["synth_path"]

    # Figures
    fig2 = os.path.join(figures_dir, "fig2_km_survival.png")
    km_curves(pd.read_parquet(synth), pd.read_parquet(synth), fig2)  # toy: same df for both

    fig3 = os.path.join(figures_dir, "fig3_cumulative_reward.png")
    fig_cumulative_reward(os.path.join(out_dir, "train_rewards.csv"), fig3)

    fig4 = os.path.join(figures_dir, "fig4_tacrolimus_patterns.png")
    fig_tacrolimus_patterns(synth, fig4)

    fig5 = os.path.join(figures_dir, "fig5_tradeoffs.png")
    bar_tradeoffs(fig5)

    print("Artifacts written to:", out_dir)
    print("Figures written to:", figures_dir)

if __name__ == "__main__":
    main()
