import numpy as np
import pandas as pd
from typing import Tuple

RNG = np.random.default_rng

def generate_synthetic_cohort(n=2000, seed=123) -> pd.DataFrame:
    """
    Create a toy longitudinal LT cohort with visits and outcomes.
    Columns:
      id, time_months, ALT, AST, ALP, ALB, creatinine, tacrolimus, event, event_time, alive
    """
    rng = RNG(seed)
    ids = np.arange(1, n+1)
    grid = np.array([0, 1, 3, 12, 60, 84, 120, 180, 240])
    rows = []
    for pid in ids:
        base_risk = rng.uniform(0.05, 0.25)
        renal_risk = rng.uniform(0.0, 0.2)
        cancer_risk = rng.uniform(0.0, 0.15)
        inf_risk = rng.uniform(0.0, 0.2)
        rej_risk = rng.uniform(0.02, 0.15)
        tac0 = rng.uniform(8.0, 12.0)
        for t in grid:
            ALT = max(5, rng.normal(40, 20))
            AST = max(5, rng.normal(35, 20))
            ALP = max(30, rng.normal(160, 60))
            ALB = max(20, rng.normal(30, 5))
            creat = max(40, rng.normal(110 + 2*renal_risk* (t/12), 20))
            tac = max(2.0, tac0 - 0.02 * t + rng.normal(0, 0.5))

            rows.append((pid, t, ALT, AST, ALP, ALB, creat, tac,
                         base_risk, renal_risk, cancer_risk, inf_risk, rej_risk))
    df = pd.DataFrame(rows, columns=[
        "id","time_months","ALT","AST","ALP","ALB","creatinine","tacrolimus",
        "base_risk","renal_risk","cancer_risk","infection_risk","rejection_risk"
    ])
    df["haz_component"] = (
        0.002*df["base_risk"] + 
        0.001*df["renal_risk"] + 0.001*df["cancer_risk"] +
        0.001*df["infection_risk"] + 0.001*df["rejection_risk"]
    ) * (1 + 0.03*np.maximum(df["tacrolimus"]-10, 0))
    haz = df.groupby("id")["haz_component"].sum().reset_index(name="cum_haz")
    haz["event_time"] = (1.0 / (1e-3 + haz["cum_haz"])) * RNG(seed).uniform(0.3, 1.7, size=len(haz))
    haz["event_time"] = haz["event_time"].clip(0, 240)
    causes = ["graft_failure", "infection", "cancer", "cardiac", "renal"]
    haz["event"] = RNG(seed+1).choice(causes + ["censored"], size=len(haz), p=[0.12,0.08,0.06,0.05,0.04,0.65])
    df = df.merge(haz[["id","event_time","event"]], on="id", how="left")
    df["alive"] = (df["time_months"] < df["event_time"]) | (df["event"]=="censored")
    return df

def load_cohort(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format")

def save_cohort(df: pd.DataFrame, path: str):
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported file format")
