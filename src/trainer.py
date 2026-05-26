"""Phase1Trainer v15 — KLD + Ranking + VICReg + PriceLoss."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from config import N_ASSETS, BATCH_SIZE


class Phase1Trainer:
    def __init__(self, model, lr=1e-3, commission=0.001, device="cpu",
                 asset_names=None, rank_weight=0.05, vic_weight=0.05):
        self.model      = model
        self.device     = device
        self.commission = commission
        self.rw         = rank_weight
        self.vw         = vic_weight
        names = asset_names or [f"coin{i}" for i in range(N_ASSETS)]
        self.names = names if names[-1] == "현금" else names + ["현금"]
        self.n_out = N_ASSETS + 1
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    @staticmethod
    def _kld(pred, target, fw):
        log_pred = torch.log(pred.clamp(1e-8))
        loss = F.kl_div(log_pred, target, reduction="none").sum(dim=-1)
        return (loss * fw).mean()

    @staticmethod
    def _rk(v_emb, target):
        regime = (target[:, -1] > 0.5).float()
        vn  = F.normalize(v_emb, dim=-1)
        sim = vn @ vn.T
        same = (regime[:, None] == regime[None, :]).float()
        return F.relu(1.0 - sim * (same * 2 - 1)).mean()

    @staticmethod
    def _vic(z, lv=1.0, lc=0.04, eps=1e-4):
        B, D = z.shape
        std = torch.sqrt(z.var(dim=0) + eps)
        vl  = F.relu(1.0 - std).mean()
        zc  = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (B - 1)
        cl  = cov.pow(2).triu(1).sum() / D
        return lv * vl + lc * cl

    def _price_loss(self, price_pred, prices, idx_b, horizon=12):
        prices_t = torch.FloatTensor(prices).to(self.device)
        losses   = []
        for i, idx in enumerate(idx_b):
            idx = int(idx.item())
            end = min(idx + horizon, len(prices) - 1)
            if end <= idx:
                continue
            fut      = prices_t[idx:end + 1]
            curr     = prices_t[idx]
            fut_rets = (fut - curr) / (curr + 1e-9)
            target   = torch.stack([fut_rets.min(dim=0).values,
                                    fut_rets.max(dim=0).values], dim=-1)
            losses.append(F.mse_loss(price_pred[i], target))
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _equity(self, X, prices):
        self.model.eval()
        e = 1.0
        prev = np.zeros(self.n_out); prev[-1] = 1.0
        wc = torch.zeros(1, self.n_out, device=self.device); wc[0, -1] = 1.0
        with torch.no_grad():
            for i in range(len(X) - 1):
                xb = torch.FloatTensor(X[i]).unsqueeze(0).to(self.device)
                w, _, _ = self.model(xb, wc)
                w = w.squeeze(0).cpu().numpy()
                w = np.clip(w, 0, 1); w /= (w.sum() + 1e-8)
                r = (prices[i + 1] - prices[i]) / (prices[i] + 1e-9)
                e *= (1 + np.dot(w[:N_ASSETS], r)
                      - self.commission * np.abs(w - prev).sum())
                prev = w
                wc = torch.FloatTensor(w).unsqueeze(0).to(self.device)
        self.model.train()
        return e - 1.0

    def train(self, X, oracle_labels, focal_weights, prices,
              n_epochs=80, batch_size=BATCH_SIZE, log_interval=20,
              save_path=None, round_name=""):
        indices = torch.arange(len(X))
        ds = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(oracle_labels),
            torch.FloatTensor(focal_weights),
            indices)
        dl  = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=n_epochs)

        btc = prices[-1, 0] / prices[0, 0] - 1
        print(f"  loss=KLD+{self.rw}*rank+{self.vw}*vic+0.3*price  BTC:{btc*100:+.2f}%")

        prev_wc = torch.zeros(1, self.n_out, device=self.device)
        prev_wc[0, -1] = 1.0
        best = float("inf")

        for ep in range(n_epochs):
            tk = tr = tv = tp = 0.0; nb = 0
            for xb, yb, fw, idx_b in dl:
                xb = xb.to(self.device); yb = yb.to(self.device)
                fw = fw.to(self.device); B = xb.shape[0]
                wc_b = prev_wc.expand(B, -1).detach()
                pred, v, price_pred = self.model(xb, wc_b)
                kld  = self._kld(pred, yb, fw)
                rk   = self._rk(v, yb)
                vic  = self._vic(v)
                pl   = self._price_loss(price_pred, prices, idx_b, horizon=12)
                loss = kld + self.rw * rk + self.vw * vic + 0.3 * pl
                if not torch.isfinite(loss):
                    continue
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    w_last, _, _ = self.model(xb[-1:], prev_wc)
                prev_wc = w_last.detach()
                tk += kld.item(); tr += rk.item()
                tv += vic.item(); tp += pl.item(); nb += 1
            sch.step()
            avg_kld = tk / max(nb, 1)
            if avg_kld < best:
                best = avg_kld
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            if (ep + 1) % log_interval == 0:
                mod_ret = self._equity(X, prices)
                wc_fix  = torch.zeros(1, self.n_out, device=self.device)
                wc_fix[0, -1] = 1.0
                with torch.no_grad():
                    sp, _, _ = self.model(
                        torch.FloatTensor(X[-1]).unsqueeze(0).to(self.device), wc_fix)
                    sp = sp.squeeze(0).cpu().numpy()
                ps = " ".join(f"{n}={sp[i]*100:.0f}%" for i, n in enumerate(self.names))
                print(f"    ep {ep+1:3d}  kld={avg_kld:.5f}  "
                      f"rk={tr/max(nb,1):.5f}  price={tp/max(nb,1):.5f}  "
                      f"v_std={v.std(dim=0).mean().item():.4f}")
                print(f"            모델:{mod_ret*100:+.2f}%  BTC:{btc*100:+.2f}%")
                print(f"            pred=[{ps}]")

        return self._equity(X, prices), btc
