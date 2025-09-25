import numpy as np
import pandas as pd
from typing import Dict, Tuple

def build_state_action_spaces(df: pd.DataFrame) -> Dict:
    # States: [ALT, AST, ALP, ALB, creatinine, tacrolimus] standard-scaled per time
    features = ["ALT","AST","ALP","ALB","creatinine","tacrolimus"]
    stats = df[features].describe().to_dict()
    # Actions: discrete delta on tac: {-1.0, -0.5, 0, +0.5, +1.0}
    actions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    return {"features": features, "stats": stats, "actions": actions}

def standardize(df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
    out = df.copy()
    for col in stats:
        if isinstance(stats[col], dict) and "mean" in stats[col] and "std" in stats[col]:
            mu = stats[col]["mean"]
            sd = stats[col]["std"] if stats[col]["std"]>0 else 1.0
            out[col] = (out[col]-mu)/sd
    return out

def make_transitions(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    # Make (s,a,r,s') tuples from longitudinal rows per patient
    feats = cfg["features"]
    actions = cfg["actions"]
    df = df.sort_values(["id","time_months"])
    rows = []
    for _, g in df.groupby("id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)-1):
            s = g.loc[i,feats].values.astype(np.float32)
            s_next = g.loc[i+1,feats].values.astype(np.float32)
            # Proxy reward (toy): +1 if alive at next, minus penalties
            alive_next = bool(g.loc[i+1,"alive"])
            r = 1.0 if alive_next else -1.0
            tac = g.loc[i,"tacrolimus"]
            r -= 0.01 * max(tac - 9.0, 0)  # penalty above 9
            if not alive_next and g.loc[i+1,"time_months"] >= g.loc[i+1,"event_time"] - 1e-6:
                cause = g.loc[i+1,"event"]
                cause_w = {"graft_failure":1.0,"infection":0.6,"cancer":0.8,"cardiac":0.5,"renal":0.7}.get(cause,0.0)
                r -= cause_w
            desired = 8.0
            delta = np.clip(desired - tac, -1.0, 1.0)
            a_idx = int(np.argmin(np.abs(actions - delta)))
            rows.append((s, a_idx, r, s_next))
    out = pd.DataFrame(rows, columns=["s","a","r","s_next"])
    return out
