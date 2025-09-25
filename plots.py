import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fig_cumulative_reward(train_csv: str, out_path: str):
    rewards = pd.read_csv(train_csv, header=None).squeeze("columns")
    cum = rewards.cumsum()
    plt.figure()
    plt.plot(cum.index, cum.values, label="iDose-LT RL")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative Risk-Adjusted Reward (Synthetic)")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def fig_tacrolimus_patterns(df_path: str, out_path: str):
    df = pd.read_parquet(df_path) if df_path.endswith(".parquet") else pd.read_csv(df_path)
    g = df.groupby("time_months")["tacrolimus"].mean()
    plt.figure()
    plt.plot(g.index, g.values, label="Average tacrolimus (synthetic)")
    plt.xlabel("Months since transplant")
    plt.ylabel("Tacrolimus (ng/mL)")
    plt.title("Tacrolimus Trajectories (Synthetic)")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def bar_tradeoffs(out_path: str):
    labels = ["Rejection","Infection","Cancer","Cardiac","Renal"]
    hist = np.array([5.0, 15.0, 4.0, 10.0, 6.0])
    rl   = np.array([3.9, 10.0, 2.0, 5.0, 3.0])
    x = np.arange(len(labels))
    w = 0.35
    plt.figure()
    plt.bar(x - w/2, hist, width=w, label="Historical")
    plt.bar(x + w/2, rl,   width=w, label="iDose-LT RL")
    plt.xticks(x, labels)
    plt.ylabel("Event probability (%)")
    plt.title("Trade-off Analysis (Toy Illustration)")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
