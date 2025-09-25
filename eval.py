import os
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from typing import Dict, Tuple
from .model import QNet
from .preprocess import build_state_action_spaces, standardize

def load_artifacts(out_dir: str, state_dim: int, action_dim: int):
    q = QNet(state_dim, action_dim)
    q.load_state_dict(torch.load(os.path.join(out_dir, "qnet.pt"), map_location="cpu"))
    q.eval()
    return q

def km_from_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    d = df.groupby("id").agg(event_time=("event_time","max"), event=("event","first"))
    d["duration"] = d["event_time"]
    d["observed"] = (d["event"]!="censored").astype(int)
    return d

def km_curves(df_hist: pd.DataFrame, df_rl: pd.DataFrame, out_path: str):
    kmf = KaplanMeierFitter()
    import matplotlib.pyplot as plt

    hist = km_from_synthetic(df_hist)
    rl = km_from_synthetic(df_rl)

    plt.figure()
    kmf.fit(hist["duration"], event_observed=hist["observed"], label="Historical")
    kmf.plot(ci_show=False, linestyle="--")
    kmf.fit(rl["duration"], event_observed=rl["observed"], label="iDose-LT RL")
    kmf.plot(ci_show=False)
    plt.xlabel("Months since transplant")
    plt.ylabel("Survival probability")
    plt.title("Overall Survival (Synthetic)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
