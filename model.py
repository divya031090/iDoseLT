import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.s = []
        self.a = []
        self.r = []
        self.s_next = []
        self._idx = 0
    def push(self, s, a, r, s_next):
        if len(self.s) < self.capacity:
            self.s.append(s); self.a.append(a); self.r.append(r); self.s_next.append(s_next)
        else:
            self.s[self._idx] = s
            self.a[self._idx] = a
            self.r[self._idx] = r
            self.s_next[self._idx] = s_next
            self._idx = (self._idx + 1) % self.capacity
    def sample(self, batch_size):
        import random
        idxs = random.sample(range(len(self.s)), batch_size)
        s = torch.tensor([self.s[i] for i in idxs], dtype=torch.float32)
        a = torch.tensor([self.a[i] for i in idxs], dtype=torch.long)
        r = torch.tensor([self.r[i] for i in idxs], dtype=torch.float32)
        s_next = torch.tensor([self.s_next[i] for i in idxs], dtype=torch.float32)
        return s,a,r,s_next
    def __len__(self):
        return len(self.s)
