"""Multi-task MLP on the NEW-FEATURES (nf) extended feature set. Mirrors run_mlp.py
exactly (same architecture / hyperparameters), only swapping common -> common_nf and
output name -> mlp_nf20, to isolate the lift from the new factors alone.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common_nf20 import load, feature_cols, evaluate, save_pred, print_row, METR

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class CrossNet(nn.Module):
    def __init__(self, d, n=2):
        super().__init__()
        self.w = nn.ModuleList([nn.Linear(d, 1, bias=True) for _ in range(n)])
    def forward(self, x0):
        x = x0
        for lin in self.w:
            x = x0 * lin(x) + x
        return x


class MTMLP(nn.Module):
    def __init__(self, d, h=384, p=0.15):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)
        self.cross = CrossNet(d, 2)
        self.tower = nn.Sequential(
            nn.Linear(d * 2, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.SiLU(), nn.Dropout(p))
        self.main = nn.Linear(h // 2, 1)
        self.sign = nn.Linear(h // 2, 1)
        self.mag = nn.Linear(h // 2, 1)
    def forward(self, x):
        x = self.bn(x)
        z = torch.cat([self.cross(x), x], -1)
        z = self.tower(z)
        return self.main(z).squeeze(-1), self.sign(z).squeeze(-1), self.mag(z).squeeze(-1)


def daily_ic_np(pred, y, day):
    d = pd.DataFrame({"p": pred, "t": y, "day": day}).groupby("day").apply(lambda s: s.p.corr(s.t))
    return float(d.mean())


def main():
    t0 = time.time()
    fcols = feature_cols()
    df = load(fcols)
    print(f"[mlp-nf20] {len(df):,} rows, {len(fcols)} feats, dev={DEV}", flush=True)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "valid").to_numpy()
    X = torch.from_numpy(df[fcols].to_numpy("float32"))
    yxs = df["y_xs"].to_numpy("float32")
    sgn = (df["y"].to_numpy() > 0).astype("float32")
    mag = np.abs(yxs).astype("float32")
    Xtr, ytr_t = X[tr].to(DEV), torch.from_numpy(yxs[tr]).to(DEV)
    sgn_t = torch.from_numpy(sgn[tr]).to(DEV); mag_t = torch.from_numpy(mag[tr]).to(DEV)
    day = df["day"].to_numpy()
    yv = df["y"].to_numpy()

    def train_one(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        net = MTMLP(len(fcols)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-4)
        n = Xtr.shape[0]; bs = 8192
        best_ic, best_state, bad = -9, None, 0
        for ep in range(40):
            net.train(); perm = torch.randperm(n, device=DEV)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                m, s, g = net(Xtr[idx])
                loss = (nn.functional.smooth_l1_loss(m, ytr_t[idx])
                        + 0.3 * nn.functional.binary_cross_entropy_with_logits(s, sgn_t[idx])
                        + 0.3 * nn.functional.mse_loss(g, mag_t[idx]))
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(X[va].to(DEV))[0].cpu().numpy()
            ic = daily_ic_np(pv, yv[va], day[va])
            if ic > best_ic + 1e-5:
                best_ic, best_state, bad = ic, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 6: break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pred = np.concatenate([net(X[i:i+200000].to(DEV))[0].cpu().numpy()
                                   for i in range(0, len(X), 200000)])
        print(f"   seed {seed}: best valid IC {best_ic:.5f} (ep stop)", flush=True)
        return pred

    seeds = (0, 1, 2, 3, 4, 5)
    preds = np.mean([train_one(s) for s in seeds], axis=0)
    df["pred"] = preds
    r = evaluate(df, "mlp_nf20"); print_row(r); save_pred(df, "mlp_nf20")
    pd.DataFrame([r]).to_csv(METR / "mlp_nf20_leaderboard.csv", index=False)
    print(f"[mlp-nf20] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
