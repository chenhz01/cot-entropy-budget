"""ceb_llmverify_v2.py — 轻量真模型复跑：每步 1 次 API 调用 + 本地前向模拟。

设计动机（v1 太慢的教训）：v1 对 beam 每状态发一次调用，12 实例 ≈ 2000 次。
v2 把「候选取舍」改为本地确定性前向模拟（像 value network），LLM 只当候选生成器：
  - 每步 1 次调用：模型给 ≤6 个候选操作；
  - 本地前向模拟：从每个候选出发，用程序化穷举看"离 24 还有多远"（可达性势分）；
  - 政策只决定保留几个候选进模拟排序：
      uniform-k：固定 k 个；
      ceb-v4：并列最优候选数 = 熵（与 simharness v4 逐字一致的语义）；
      greedy：只取模型第一个候选（pass@1 单链基线）。
指标：pass@1（解出率）、平均步数、API 调用与 token（全成本口径）。

用法：
  python ceb_llmverify_v2.py --models-json <path> --instances 8 --max-calls 200
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ceb_simharness as H  # 复用程序化穷举（_solvable 势分 + gen_instances）

DEFAULT_MODELS_JSON = os.environ.get("CEB_MODELS_JSON", "")
OP_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*(?:=|得|→)\s*(-?\d+(?:\.\d+)?)\s*$")

SYS = ("你是 24 点游戏求解助手。每次只能选两个数做 + - * /，用掉这两个数、"
       "把结果放回集合，直到出现 24。每行输出一个候选操作，格式严格为："
       "a op b = r（如 8 * 3 = 24）。不要解释，只列候选。")


def chat(url, key, model, messages, temperature, max_tokens=1500):
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature,
                       "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.load(resp)
    usage = data.get("usage", {})
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip():
        content = msg.get("reasoning_content") or ""
    return content, usage


def parse_ops(text, pool):
    ops, seen = [], set()
    for line in text.splitlines():
        m = OP_RE.match(line.strip().strip("`"))
        if not m:
            continue
        fa, op, fb, fr = (Fraction(m.group(1)), m.group(2),
                          Fraction(m.group(3)), Fraction(m.group(4)))
        if fa not in pool or fb not in pool:
            continue
        calc = {"+": fa + fb, "-": fa - fb, "*": fa * fb,
                "/": (fa / fb if fb != 0 else None)}[op]
        if calc is None or calc != fr:
            continue
        k = (fa, op, fb)
        if k not in seen:
            seen.add(k)
            ops.append((fa, op, fb, fr))
    return ops


def gen_candidates(url, key, model, pool, temp, tracker):
    prompt = (f"当前数字集合：{[str(x) for x in pool]}\n"
              f"给出最多 6 个候选操作（每行 a op b = r）：")
    try:
        content, usage = chat(url, key, model,
                              [{"role": "system", "content": SYS},
                               {"role": "user", "content": prompt}], temp)
    except Exception as e:
        tracker["errors"].append(str(e)[:150])
        return []
    tracker["calls"] += 1
    tracker["in_tok"] += usage.get("prompt_tokens", 0)
    tracker["out_tok"] += usage.get("completion_tokens", 0)
    return parse_ops(content, pool)


def score_op(op):
    r, o = op[3], op[1]
    return (int(r.denominator == 1), int(1 <= abs(r) <= 100),
            {"+": 3, "*": 2, "-": 1, "/": 0}[o])


def potential(pool):
    """本地前向模拟势分：程序化穷举看该状态可达 24 的最短步数（-1=不可达）。"""
    if Fraction(24) in pool:
        return 0
    for op in H._candidates(tuple(sorted(pool))):
        newpool = [x for x in pool if x not in (op[0], op[1])] + [op[3]]
        sub = potential(newpool)
        if sub >= 0:
            return 1 + sub
    return -1


def width_of(policy, cands):
    if not cands:
        return 0
    if policy == "greedy":
        return 1
    if policy == "uniform-4":
        return 4
    if policy == "ceb-v4":
        top = max(score_op(c) for c in cands)
        ties = sum(1 for c in cands if score_op(c) == top)
        return max(1, min(ties, 4))
    raise ValueError(policy)


def solve(nums, target, policy, url, key, model, tracker, temp=0.7, max_steps=8):
    pool = list(nums)
    steps = 0
    while steps < max_steps:
        if target in pool:
            return True, steps
        cands = gen_candidates(url, key, model, pool, temp, tracker)
        if not cands:
            return False, steps
        ranked = sorted(cands, key=score_op, reverse=True)
        w = width_of(policy, ranked)
        kept = ranked[:w]
        # 本地前向模拟：势分最小者胜（并列取启发分高者）
        best, best_key = None, None
        for op in kept:
            newpool = [x for x in pool if x not in (op[0], op[1])] + [op[3]]
            pot = potential(newpool)
            key = (pot if pot >= 0 else 99, [-x for x in score_op(op)])
            if best_key is None or key < best_key:
                best, best_key = op, key
        if best is None:
            return False, steps
        pool = [x for x in pool if x not in (best[0], best[1])] + [best[3]]
        steps += 1
    return target in pool, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-json", default=DEFAULT_MODELS_JSON)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--policies", default="greedy,uniform-4,ceb-v4")
    ap.add_argument("--out", default="llmverify_v2.json")
    args = ap.parse_args()

    if not args.models_json:
        raise SystemExit("请用 --models-json 或 CEB_MODELS_JSON 指定 models 配置")
    with io.open(args.models_json, encoding="utf-8") as f:
        models = {m["id"]: m for m in json.load(f)}
    if args.model not in models:
        raise SystemExit(f"models 配置无 {args.model}")
    m = models[args.model]
    url, key = m["url"], m["apiKey"]

    insts = [i for i in H.gen_instances(args.seed, args.instances)
             if i.target is not None]
    policies = args.policies.split(",")
    tracker = {"calls": 0, "in_tok": 0, "out_tok": 0, "errors": []}
    res = {p: {"solved": 0, "steps": 0, "n": 0} for p in policies}
    t0 = time.time()
    for idx, inst in enumerate(insts):
        if tracker["calls"] >= args.max_calls:
            print(f"[预算闸] 停于实例 {idx}/{len(insts)}")
            break
        nums = [Fraction(x) for x in inst.nums]
        line = []
        for p in policies:
            if tracker["calls"] >= args.max_calls:
                line.append(f"{p}=ABORT")
                continue
            ok, st = solve(nums, inst.target, p, url, key, args.model,
                           tracker)
            res[p]["n"] += 1
            res[p]["solved"] += int(ok)
            res[p]["steps"] += st
            line.append(f"{p}={'✓' if ok else '✗'}({st}步)")
        print(f"  实例{idx} {[str(x) for x in inst.nums]}→{inst.target}: "
              + " ".join(line) + f" | 累计调用 {tracker['calls']}")

    out = {"model": args.model, "task": "24点(前向模拟版)", "seed": args.seed,
           "policies": {}, "cost": dict(tracker),
           "wall_sec": round(time.time() - t0, 1)}
    for p in policies:
        r = res[p]
        n = max(1, r["n"])
        out["policies"][p] = {
            "solved": r["solved"], "n": r["n"],
            "pass1": round(r["solved"] / n, 4),
            "avg_steps": round(r["steps"] / n, 2)}
    with io.open(os.path.join(HERE, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"→ {args.out} 落盘")


if __name__ == "__main__":
    main()
