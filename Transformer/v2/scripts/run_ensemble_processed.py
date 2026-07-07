"""EG ML ensemble PROCESSED (diversity-weighted + group-g-neutralized blend).
Generic version of ML_ensemble/scripts/run_ensemble_v3factors_v2.py,
parametrized by RETRAIN_TAG."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_generic import daily_ic, PREDS, METR, ROOT, TAG

ALPHA_GRID = np.round(np.arange(0.0, 0.86, 0.05), 2)
PLATEAU = 0.997


def per_day_z(df, col):
    g = df.groupby("day")[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-9)


def family_of(stem: str):
    s = stem.lower()
    if not s.endswith(f"_{TAG}"):
        return None
    if s.startswith("lightgbm") or s.startswith("lgb"):
        return "lightgbm"
    if s.startswith("mlp"):
        return "mlp"
    if s.startswith("transformer"):
        return "transformer"
    return None


def load_bases():
    files = {
        f.stem: f
        for f in PREDS.glob(f"*_{TAG}.parquet")
        if f.stem.startswith(("lightgbm_", "mlp_", "transformer_"))
    }
    fams: dict = {}
    for stem in files:
        fam = family_of(stem)
        if fam:
            fams.setdefault(fam, []).append(stem)

    base = None
    valic = {}
    for stem, f in sorted(files.items()):
        if family_of(stem) is None:
            continue
        d = pd.read_parquet(f).rename(columns={"pred": stem})
        d[stem] = per_day_z(d, stem)
        cols = ["day", "instrument_id", stem] + (["split", "y"] if base is None else [])
        base = d[cols] if base is None else base.merge(
            d[["day", "instrument_id", stem]], on=["day", "instrument_id"], how="inner")
    va = base[base.split == "valid"]
    for stem in [c for c in base.columns if c not in ("day", "instrument_id", "split", "y")]:
        valic[stem] = daily_ic(va.assign(pred=va[stem]))[0]

    members = sorted(fams.get("lightgbm", []))
    for fam in ("mlp", "transformer"):
        cands = [c for c in fams.get(fam, []) if c in base.columns]
        if cands:
            members.append(max(cands, key=lambda c: valic[c]))
    members = [m for m in members if m in base.columns]
    return base, members, valic, fams


def diversity_weights(base, members):
    C = base[base.split == "valid"][members].corr().to_numpy()
    w = np.clip(np.linalg.pinv(C).sum(1), 0, None)
    if w.sum() == 0:
        w = np.ones(len(members))
    return w / w.sum()


def neutralize_g(df, srccol, alpha, gcol="g"):
    gm = df.groupby(["day", gcol])[srccol].transform("mean")
    tmp = pd.DataFrame({"day": df["day"], "r": df[srccol] - alpha * gm})
    g = tmp.groupby("day")["r"]
    return (tmp["r"] - g.transform("mean")) / (g.transform("std") + 1e-9)


def wdict(keys, w):
    return {k: round(float(x), 3) for k, x in zip(keys, w)}


def ic_ir(base, col):
    out = {}
    for sp in ("valid", "test"):
        sub = base[base.split == sp]
        ic, s, ir = daily_ic(sub.assign(pred=sub[col]))
        out[f"{sp}_IC"], out[f"{sp}_IR"] = round(ic, 5), round(ir, 3)
    return out


def existing_score(path, label):
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    row = {"model": label}
    for sp in ("valid", "test"):
        sub = d[d.split == sp]
        ic, s, ir = daily_ic(sub)
        row[f"{sp}_IC"], row[f"{sp}_IR"] = round(ic, 5), round(ir, 3)
    return row


def main():
    base, members, valic, fams = load_bases()
    g = pd.read_parquet(ROOT / "artifacts" / "features.parquet", columns=["day", "instrument_id", "g"])
    base = base.merge(g, on=["day", "instrument_id"], how="left")
    print(f"[ens-{TAG}] families: {fams}", flush=True)
    print(f"[ens-{TAG}] blend members: {members}", flush=True)
    print(f"[ens-{TAG}] valid IC per candidate: { {k: round(v,5) for k,v in valic.items()} }", flush=True)

    M = base[members].to_numpy()
    rows = []

    fam3 = [c for c in ["lightgbm", "mlp", "transformer"] if c in base.columns]
    if len(fam3) < 3:
        lgb_vars = [m for m in members if family_of(m) == "lightgbm"]
        base["lightgbm"] = base[lgb_vars].mean(axis=1)
        best_mlp = max([m for m in members if family_of(m) == "mlp"], key=lambda c: valic[c])
        best_tr = max([m for m in members if family_of(m) == "transformer"], key=lambda c: valic[c])
        fam3 = ["lightgbm", best_mlp, best_tr]
    base["m_equal"] = base[fam3].to_numpy().mean(1)
    rows.append({"model": f"ref_equal3_{TAG}", **ic_ir(base, "m_equal"), "detail": f"equal {fam3}"})

    w_div3 = diversity_weights(base, fam3)
    base["m_div3"] = (base[fam3].to_numpy() * w_div3).sum(1)
    rows.append({"model": f"div3_{TAG}", **ic_ir(base, "m_div3"), "detail": f"w={wdict(fam3, w_div3)}"})

    w = diversity_weights(base, members)
    base["m_div"] = (M * w).sum(1)
    rows.append({"model": f"divN_{TAG}", **ic_ir(base, "m_div"), "detail": f"w={wdict(members, w)}"})

    valid_ic_by_alpha = {}
    for a in ALPHA_GRID:
        col = f"_na_{a}"
        base[col] = neutralize_g(base, "m_div", a)
        sub = base[base.split == "valid"]
        valid_ic_by_alpha[a] = daily_ic(sub.assign(pred=sub[col]))[0]
    vmax = max(valid_ic_by_alpha.values())
    alpha = float(min(a for a in ALPHA_GRID if valid_ic_by_alpha[a] >= PLATEAU * vmax))
    print(f"[ens-{TAG}] valid IC max {vmax:.5f} @plateau -> selected neutralise alpha = {alpha}", flush=True)

    base["pred"] = neutralize_g(base, "m_div", alpha)
    final = ic_ir(base, "pred")
    rows.append({"model": f"FINAL ensemble_{TAG} (divN+neutG)", **final,
                 "detail": f"alpha={alpha}  members={members}  w={wdict(members, w)}"})

    lb = pd.DataFrame(rows)
    print("\n" + lb.to_string(index=False), flush=True)
    out_lb = ROOT / "ML_ensemble" / "metrics" / f"ensemble_{TAG}_leaderboard.csv"
    out_lb.parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out_lb, index=False)

    out = base[["day", "instrument_id", "split", "y", "pred"]].copy()
    out.to_parquet(PREDS / f"ensemble_{TAG}.parquet", index=False)
    print(f"\n[ens-{TAG}] FINAL  valid IC {final['valid_IC']:.5f} IR {final['valid_IR']:.3f} | "
          f"test IC {final['test_IC']:.5f} IR {final['test_IR']:.3f}", flush=True)

    print(f"\n[ens-{TAG}] === comparison vs. existing baselines ===", flush=True)
    for path, label in [(PREDS / "ensemble_final.parquet", "ensemble_final (original 213 feat)"),
                         (PREDS / "ensemble_v2.parquet", "ensemble_v2 (best prior, 213 feat)")]:
        row = existing_score(path, label)
        if row:
            print(f"  {label:38s} valid IC {row['valid_IC']:.5f} IR {row['valid_IR']:.3f} | "
                  f"test IC {row['test_IC']:.5f} IR {row['test_IR']:.3f}", flush=True)
    print(f"  {f'ensemble_{TAG} (213+new factors)':38s} valid IC {final['valid_IC']:.5f} IR {final['valid_IR']:.3f} | "
          f"test IC {final['test_IC']:.5f} IR {final['test_IR']:.3f}", flush=True)

    json.dump({"members": members, "weights": w.tolist(), "neut_alpha": alpha},
              open(ROOT / "ML_ensemble" / "models" / f"ensemble_{TAG}_config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
