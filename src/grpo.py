"""GRPO Phase 2 trainer  v15."""
import numpy as np
import torch
from torch.distributions import Dirichlet
from config import N_ASSETS, COMMISSION, GRPO_G, GRPO_CONC, GRPO_LR, BATCH_SIZE


def manual_update(model, loss, lr, clip=0.5):
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.sub_(lr * p.grad.clamp(-clip, clip))
    model.zero_grad()


def grpo_step(model, xb, wc, p_curr, p_next, device,
              g=GRPO_G, conc=GRPO_CONC):
    with torch.no_grad():
        alpha, _, _ = model(xb, wc)
    c    = (alpha.detach() * conc + 1e-3).clamp(1e-3)
    dist = Dirichlet(c)
    ws   = torch.stack([dist.sample() for _ in range(g)], dim=1)  # (B,G,6)

    rets   = torch.FloatTensor((p_next - p_curr) / (p_curr + 1e-9)).to(device)
    port_r = (ws[:, :, :N_ASSETS] * rets.unsqueeze(1)).sum(-1)
    cost   = COMMISSION * (ws - wc.unsqueeze(1)).abs().sum(-1)
    reward = port_r - cost  # pure reward — no asymmetric penalty

    adv    = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-8)
    a2, _, _ = model(xb, wc)
    c2     = (a2 * conc + 1e-3).clamp(1e-3)
    lp     = Dirichlet(c2).log_prob(ws.detach().permute(1, 0, 2)).T
    return -(adv.detach() * lp).mean(), reward.mean().item()


def train_grpo(model, X_tr, P_tr, device,
               n_epochs=200, lr=GRPO_LR, batch_size=BATCH_SIZE,
               g=GRPO_G, conc=GRPO_CONC):
    for ep in range(n_epochs):
        model.train()
        prev_wc = torch.zeros(1, N_ASSETS + 1, device=device); prev_wc[0, -1] = 1.0
        tl = tr = nb = 0.0
        for s in range(0, len(X_tr) - 1, batch_size):
            bi = np.arange(s, min(s + batch_size, len(X_tr) - 1))
            if len(bi) < 4:
                continue
            xb   = torch.FloatTensor(X_tr[bi]).to(device)
            wc   = prev_wc.expand(len(bi), -1).clone()
            loss, r = grpo_step(model, xb, wc, P_tr[bi], P_tr[bi + 1], device, g, conc)
            if not torch.isfinite(loss):
                continue
            manual_update(model, loss, lr)
            tl += loss.item(); tr += r; nb += 1
            with torch.no_grad():
                w_out, _, _ = model(xb[-1:], prev_wc)
            prev_wc = w_out.detach()
        if (ep + 1) % 20 == 0:
            print(f"    ep {ep+1:3d}  loss={tl/max(nb,1):.5f}  avg_r={tr/max(nb,1):.6f}")
