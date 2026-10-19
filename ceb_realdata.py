"""CEB-REALRUN · 真语料推理链上的熵预算分布跑批（卫星件 2 · Step 2.75）

Step 2.8 原则：真数据先查本地，不急着联网。本器读取真实推理链文本：
  1) 勾选卡式表格的推演链列（第 5 列，→ 分隔）
  2) 列表式决策记录（- 开头的条目）
按 → 切分为推理步，跑 CEB 政策，输出熵档分布 JSON。

用法：
  python ceb_realdata.py                              # 用包内 examples/reasoning_chains.md
  python ceb_realdata.py --files a.md b.md            # 用任意语料（表格/列表混排）

诚实边界 [推算]：文本档位判定是结构信号启发式，不是模型不确定性；
本输出证明的是"政策在真实推理文本上的行为分布"，不是 LLM 增益。
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ceb_policy import Step, policy_tier, HIGH, MED, LOW  # noqa: E402

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "examples", "reasoning_chains.md")


def extract_chain_cells(path):
    """从勾选卡表格行里抠出『推演链』列（第 5 列）的文本；过滤表头与说明行。"""
    cells = []
    if not os.path.exists(path):
        return cells
    for line in open(path, encoding="utf-8"):
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 5:
            continue
        # 表头行/说明行过滤：子步列是"子步"或以"#"开头，或推演链列是列名本身
        if cols[0] in ("#", "") or cols[1] in ("子步", "") or cols[4].startswith("推演链（"):
            continue
        if "→" in cols[4] and "---" not in cols[4]:
            cells.append(cols[4])
    return cells


def extract_memory_lines(path):
    if not os.path.exists(path):
        return []
    return [l.strip("- ").strip() for l in open(path, encoding="utf-8")
            if l.strip().startswith("-") and len(l.strip()) > 12]


def segment(cells):
    """把单元格按 → 切成推理步，保留原文片段。"""
    steps = []
    for cell in cells:
        parts = [p.strip() for p in re.split(r"→|->", cell) if p.strip()]
        steps.extend(parts)
    return steps


def run(corpus_paths=None):
    corpus_paths = corpus_paths or [DEFAULT_CORPUS]
    all_steps, per_file = [], {}
    for path in corpus_paths:
        name = os.path.basename(path)
        # 按内容判断文体（不依赖特定文件名）：表格推演链列 + 列表决策记录都收
        texts = segment(extract_chain_cells(path)) + extract_memory_lines(path)
        tiers = []
        for i, t in enumerate(texts):
            step = Step(idx=i, text=t[:300])
            tier, _ = policy_tier(step)
            tiers.append(tier)
            all_steps.append((name, tier, t[:120]))
        per_file[name] = {
            "steps": len(texts),
            "HIGH": tiers.count(HIGH), "MED": tiers.count(MED), "LOW": tiers.count(LOW),
        }
    total = len(all_steps)
    dist = {HIGH: 0, MED: 0, LOW: 0}
    for _, tier, _ in all_steps:
        dist[tier] += 1
    return {
        "meta": {"corpus": [os.path.basename(p) for p in corpus_paths],
                 "total_steps": total, "note": "文本档位=结构信号启发 [推算]"},
        "distribution": {k: {"n": dist[k],
                             "pct": round(dist[k] / total * 100, 1) if total else 0}
                         for k in (HIGH, MED, LOW)},
        "per_file": per_file,
        "samples": [{"file": f, "tier": t, "text": x} for f, t, x in all_steps[:8]],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None,
                    help="语料文件路径（表格/列表文体混排可）")
    args = ap.parse_args()
    out = run(args.files)
    with open("ceb_realdata.out.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("[写出] ceb_realdata.out.json")
