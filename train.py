import os, math, yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from .data import generate_synthetic_cohort, save_cohort, load_cohort
from .preprocess import build_state_action_spaces, standardize, make_transitions
from .model import QNet, ReplayBuffer

def epsilon_schedule(step, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    ratio = min(1.0, step / decay_steps)
    return start + (end - start) * ratio

def train(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # Data
    synth_path = cfg["paths"]["synth_path"]
    if not os.path.exists(synth_path):
        df = generate_synthetic_cohort(n=2000, seed=cfg["seed"])
        save_cohort(df, synth_path)
    df = load_cohort(synth_path)

    # State space & transitions
    spaces = build_state_action_spaces(df)
    df_std = df.copy()
    df_std[spaces["features"]] = standardize(df[spaces["features"]], {k: spaces["stats"][k] for k in spaces["features"]})[spaces["features"]]
    transitions = make_transitions(df_std, {"features": spaces["features"], "actions": spaces["actions"]})

    state_dim = len(spaces["features"])
    action_dim = len(spaces["actions"])

    # DQN
    q = QNet(state_dim, action_dim)
    q_tgt = QNet(state_dim, action_dim)
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=cfg["training"]["lr"])
    loss_fn = nn.SmoothL1Loss()

    buf = ReplayBuffer(capacity=cfg["training"]["buffer_size"])
    for _, row in transitions.iterrows():
        buf.push(row["s"], int(row["a"]), float(row["r"]), row["s_next"])

    eps = cfg["training"]["epsilon_start"]
    global_step = 0

    episodes = cfg["training"]["episodes"]
    steps_per_ep = cfg["training"]["max_steps_per_episode"]
    gamma = cfg["training"]["gamma"]
    target_update_every = cfg["training"]["target_update_every"]
    batch_size = cfg["training"]["batch_size"]
    start_train_after = cfg["training"]["start_train_after"]
    eps_end = cfg["training"]["epsilon_end"]
    eps_decay = cfg["training"]["epsilon_decay_steps"]

    # Simple loop over buffer as pseudo-episodes
    rewards = []
    for ep in trange(episodes, desc="Training iDose-LT DQN"):
        ep_reward = 0.0
        for step in range(steps_per_ep):
            global_step += 1
            eps = epsilon_schedule(global_step, cfg["training"]["epsilon_start"], eps_end, eps_decay)

            if len(buf) >= start_train_after:
                s,a,r,s_next = buf.sample(batch_size)
                with torch.no_grad():
                    q_next = q_tgt(s_next).max(1).values
                    y = r + gamma * q_next
                q_pred = q(s).gather(1, a.view(-1,1)).squeeze(1)
                loss = loss_fn(q_pred, y)
                opt.zero_grad(); loss.backward(); opt.step()

                if global_step % target_update_every == 0:
                    q_tgt.load_state_dict(q.state_dict())

                ep_reward += float(r.mean().item())
        rewards.append(ep_reward)

    # Save artifacts
    torch.save(q.state_dict(), os.path.join(out_dir, "qnet.pt"))
    import json
    with open(os.path.join(out_dir, "spaces.json"), "w") as f:
        f.write(json.dumps({
            "features": spaces["features"],
            "actions": spaces["actions"].tolist(),
            "stats": spaces["stats"]
        }, indent=2))
    pd.Series(rewards).to_csv(os.path.join(out_dir, "train_rewards.csv"), index=False)

    return {"out_dir": out_dir, "figures_dir": figures_dir, "synth_path": synth_path}

if __name__ == "__main__":
    train()
