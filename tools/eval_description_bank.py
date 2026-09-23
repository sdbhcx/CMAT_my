# -*- coding: utf-8 -*-
"""用信息瓶颈的两把标尺评估描述池质量（Aff3DFunc, MM'25）。

对冻结的 RoBERTa 编码出的描述嵌入计算：

  V  类内方差    同一 affordance 内描述两两余弦距离的均值 -> 多样性
  U  类间可分性  silhouette 系数 (b-a)/max(a,b) 的均值     -> 区分度
  Score = alpha * V_norm + (1 - alpha) * U_norm

两个实现要点：
  1. **全局中心化**。RoBERTa 的 mean-pooled 嵌入有各向异性，所有句子挤在
     一个窄锥里，余弦相似度普遍接近 1，直接用会让类间距离小于类内距离、
     U 变成大负数。先减去参与比较的全部描述的全局均值再归一化。
  2. **U 用 silhouette 而非 1-within/between**。前者是有界的标准可分性
     度量（[-1,1]，0 表示无结构），后者在各向异性下会失真。

V_norm / U_norm 是池内 min-max 归一化，用于定位最弱类别；跨池比较看原始值。

用法（项目根）：
    python tools/eval_description_bank.py --bank configs/description_bank_v2.json \
        --compare configs/description_bank_v1.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_MODEL = "/home/junbo/wyn/model/roberta-base"
FLAT_VIEW = "flat"


# ---------------------------------------------------------------------------
# 描述池读取：兼容 v1（entries 为 list，取 descriptions）与
#             v2（entries 为 dict，按四视角分组）
# ---------------------------------------------------------------------------

def load_bank(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    entries = raw["entries"]

    def clean(items):
        return [s.strip() for s in items if s and s.strip()]

    bank = {}
    if isinstance(entries, dict):
        for key, val in entries.items():
            if isinstance(val, dict):
                bank[key] = {v: clean(d) for v, d in val.items()}
            else:
                bank[key] = {FLAT_VIEW: clean(val)}
    else:
        for e in entries:
            key = e.get("query_key")
            bank[key] = {FLAT_VIEW: clean(e.get("descriptions", []))}

    views = []
    for per in bank.values():
        for v in per:
            if v not in views:
                views.append(v)
    return bank, views, raw.get("version", os.path.basename(path))


def flatten(bank, views=None):
    out = []
    for aff in sorted(bank.keys()):
        for view, items in bank[aff].items():
            if views is not None and view not in views:
                continue
            for s in items:
                out.append((aff, view, s))
    return out


# ---------------------------------------------------------------------------
# 编码 + 全局中心化
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode(texts, model, tokenizer, device, batch_size=32, max_length=64):
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        last = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        outs.append(pooled.cpu())
    return torch.cat(outs, 0).numpy()


def center_and_normalize(X, mu):
    Xc = X - mu
    return Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# 两把标尺
# ---------------------------------------------------------------------------

def pairwise_cosine_distance(e):
    n = e.shape[0]
    if n < 2:
        return 0.0
    sim = e @ e.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))


def silhouette_score(emb_by_aff):
    """标准 silhouette，基于余弦距离。返回均值，范围 [-1, 1]。"""
    labels, rows = [], []
    for aff in sorted(emb_by_aff.keys()):
        for r in emb_by_aff[aff]:
            rows.append(r)
            labels.append(aff)
    X = np.asarray(rows)
    labels = np.asarray(labels)
    n = X.shape[0]
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)

    uniq = sorted(set(labels.tolist()))
    idx = {u: np.where(labels == u)[0] for u in uniq}
    s = np.zeros(n)
    for i in range(n):
        own = idx[labels[i]]
        a = D[i, own].sum() / max(len(own) - 1, 1)
        b = min(D[i, idx[u]].mean() for u in uniq if u != labels[i])
        denom = max(a, b)
        s[i] = 0.0 if denom < 1e-12 else (b - a) / denom
    return float(s.mean())


def bank_metrics(emb_by_aff):
    within = {a: pairwise_cosine_distance(e) for a, e in emb_by_aff.items()}
    V = float(np.mean(list(within.values())))
    U = silhouette_score(emb_by_aff)
    return V, U, within


def minmax(d):
    ks = list(d.keys())
    vs = np.array([d[k] for k in ks], dtype=float)
    lo, hi = vs.min(), vs.max()
    if hi - lo < 1e-12:
        return {k: 0.5 for k in ks}
    return {k: float((d[k] - lo) / (hi - lo)) for k in ks}


def score_table(per_aff, alpha=0.5):
    V = {a: m["V"] for a, m in per_aff.items()}
    U = {a: m["U"] for a, m in per_aff.items()}
    Vn, Un = minmax(V), minmax(U)
    return {a: {"V": V[a], "U": U[a],
                "Score": alpha * Vn[a] + (1 - alpha) * Un[a]}
            for a in per_aff}


# ---------------------------------------------------------------------------
# 池大小曲线
# ---------------------------------------------------------------------------

def size_curve(bank_emb, sizes, repeats=5, seed=0):
    """bank_emb: {aff: [n_aff, d]}（已中心化）。返回 {M: (V, U)}"""
    rng = np.random.RandomState(seed)
    out = {}
    for M in sizes:
        Vs, Us = [], []
        for _ in range(repeats):
            sub = {}
            for aff, e in bank_emb.items():
                n = e.shape[0]
                if n <= M:
                    sub[aff] = e
                else:
                    idx = rng.choice(n, M, replace=False)
                    sub[aff] = e[idx]
            v, u, _ = bank_metrics(sub)
            Vs.append(v)
            Us.append(u)
        out[M] = (float(np.mean(Vs)), float(np.mean(Us)))
    return out


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def evaluate(bank, views, emb_lookup, mu, tag, alpha=0.5, do_curve=True):
    """emb_lookup: {text: embedding(raw)}；内部做中心化后再算指标。"""
    emb_by_aff, per_view_emb = {}, {v: {} for v in views}
    for aff in sorted(bank.keys()):
        chunks, per_view_chunks = [], {v: [] for v in views}
        for view, items in bank[aff].items():
            for s in items:
                raw = emb_lookup[s]
                chunks.append(raw)
                per_view_chunks[view].append(raw)
        emb_by_aff[aff] = center_and_normalize(np.asarray(chunks), mu)
        for v in views:
            if per_view_chunks[v]:
                per_view_emb[v][aff] = center_and_normalize(
                    np.asarray(per_view_chunks[v]), mu)

    V, U, within = bank_metrics(emb_by_aff)
    table = score_table({a: {"V": w, "U": w * 0} for a, w in within.items()},
                        alpha)
    # per-affordance 的 U 单独算（silhouette 是全局量，这里用 leave-one-out 近似：
    # 该类 vs 其最近邻类的可分性）
    for aff in within:
        a = within[aff]
        others = [o for o in bank if o != aff]
        b = float(np.mean([
            1.0 - float(emb_by_aff[aff].mean(0) @ emb_by_aff[o].mean(0))
            for o in others]))
        table[aff]["U"] = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0
    table = score_table(table, alpha)

    n_texts = sum(len(v) for per in bank.values() for v in per.values())
    print("\n===== %s =====" % tag)
    print("描述数 %d，affordance %d 个，视角 %s" % (n_texts, len(bank), views))
    print("类内方差 V = %.4f" % V)
    print("类间可分性 U (silhouette) = %.4f" % U)

    print("\n-- 分视角 --")
    for v in views:
        if not per_view_emb.get(v):
            continue
        vv, uu, _ = bank_metrics(per_view_emb[v])
        print("  %-12s V=%.4f  U=%.4f" % (v, vv, uu))

    order = sorted(table.items(), key=lambda kv: kv[1]["Score"])
    print("\n-- Score 最低 5 类 --")
    for a, m in order[:5]:
        print("  %-12s V=%.4f U=%.4f Score=%.4f" % (a, m["V"], m["U"], m["Score"]))
    print("-- Score 最高 5 类 --")
    for a, m in order[-5:]:
        print("  %-12s V=%.4f U=%.4f Score=%.4f" % (a, m["V"], m["U"], m["Score"]))

    curve = None
    if do_curve:
        max_n = min(e.shape[0] for e in emb_by_aff.values())
        sizes = tuple(s for s in (2, 4, 6, 8, 12) if s <= max_n)
        if len(sizes) >= 2:
            curve = size_curve(emb_by_aff, sizes)
            print("\n-- 池大小曲线（每档 5 次取均值）--")
            for M in sorted(curve):
                print("  M=%-3d V=%.4f  U=%.4f" % (M, curve[M][0], curve[M][1]))

    return {"tag": tag, "n_texts": n_texts, "V": V, "U": U,
            "per_aff": table, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="configs/description_bank_v2.json")
    ap.add_argument("--compare", default="configs/description_bank_v1.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default="results/description_bank_report.md")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(args.device).eval()

    bank, views, ver = load_bank(args.bank)
    banks = [(bank, views, "%s (%s)" % (os.path.basename(args.bank), ver))]

    if args.compare and os.path.exists(args.compare):
        cb, cv, cver = load_bank(args.compare)
        banks.insert(0, (cb, cv, "%s (%s)" % (os.path.basename(args.compare), cver)))

    # 一次性编码所有待比较的文本，保证中心化基准一致
    all_texts = []
    for b, _, _ in banks:
        for _, _, s in flatten(b):
            if s not in all_texts:
                all_texts.append(s)
    print("待编码描述 %d 条" % len(all_texts))
    raw = encode(all_texts, model, tokenizer, args.device)
    mu = raw.mean(0, keepdims=True)
    emb_lookup = {t: raw[i] for i, t in enumerate(all_texts)}

    results = [evaluate(b, v, emb_lookup, mu, tag, args.alpha)
               for b, v, tag in banks]

    if len(results) == 2:
        old, new = results[0], results[1]
        print("\n===== 对比：%s -> %s =====" % (old["tag"], new["tag"]))
        print("V: %.4f -> %.4f  (%+.4f)" % (old["V"], new["V"], new["V"] - old["V"]))
        print("U: %.4f -> %.4f  (%+.4f)" % (old["U"], new["U"], new["U"] - old["U"]))

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 描述池质量报告（信息瓶颈两把标尺）\n\n")
        f.write("编码模型：%s（冻结）· 已做全局中心化\n\n" % args.model)
        for r in results:
            f.write("## %s\n\n" % r["tag"])
            f.write("- 描述数：%d\n" % r["n_texts"])
            f.write("- 类内方差 V（多样性）：**%.4f**\n" % r["V"])
            f.write("- 类间可分性 U（silhouette，区分度）：**%.4f**\n" % r["U"])
            if r["curve"]:
                f.write("\n| M | V | U |\n|---|---|---|\n")
                for M in sorted(r["curve"]):
                    f.write("| %d | %.4f | %.4f |\n" % (M, *r["curve"][M]))
            f.write("\n| affordance | V | U | Score |\n|---|---|---|---|\n")
            for a in sorted(r["per_aff"]):
                m = r["per_aff"][a]
                f.write("| %s | %.4f | %.4f | %.4f |\n"
                        % (a, m["V"], m["U"], m["Score"]))
            f.write("\n")
        if len(results) == 2:
            f.write("## 结论\n\n")
            f.write("%s -> %s：V %+.4f，U %+.4f\n"
                    % (results[0]["tag"], results[1]["tag"],
                       results[1]["V"] - results[0]["V"],
                       results[1]["U"] - results[0]["U"]))

    csv_path = out_path.replace(".md", ".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bank", "affordance", "V", "U", "Score"])
        for r in results:
            for a in sorted(r["per_aff"]):
                m = r["per_aff"][a]
                w.writerow([r["tag"], a, "%.4f" % m["V"], "%.4f" % m["U"],
                            "%.4f" % m["Score"]])
    print("\n报告：%s\n明细：%s" % (out_path, csv_path))


if __name__ == "__main__":
    main()
