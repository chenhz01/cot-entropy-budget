"""ceb_llmverify.py — 真模型复跑：CEB v4 vs uniform 在 24 点任务上的 pass@1/步数/成本。

诚实边界（写入 README 与日报）：
- 这是「仿真 harness 的 LLM 迁移验证」，不是「CEB 提升 LLM 推理」的普适证明；
- 任务单一（24 点算术推理），模型单一（DeepSeek flash），样本受预算闸限制；
- 指标：greedy pass@1（单链一次成功率）、政策解出率（beam 搜索后）、
  平均步数、节点数、API 调用数与 token 成本（成本含候选生成，修掉仿真只数树边的漏洞）。

用法：
  python ceb_llmverify.py --instances 12 --max-calls 500 --out llmverify.json
  python ceb_llmverify.py --probe           # 只跑 1 次调用验证 API 通不通
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
# models 配置（含 API key）一律外置：--models-json 参数或 CEB_MODELS_JSON 环境变量，
# 不写死任何本机路径（OpenCore 脱敏纪律）。
DEFAULT_MODELS_JSON = os.environ.get("CEB_MODELS_JSON", "")

OP_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*(?:=|得|→)\s*(-?\d+(?:\.\d+)?)\s*$")


def load_api(model_key="deepseek-v4-flash", models_json=None):
    path = models_json or DEFAULT_MODELS_JSON
    if not path:
        raise SystemExit("请用 --models-json <路径> 或环境变量 CEB_MODELS_JSON "
                         "指定 OpenAI 兼容 models 配置（含 apiKey/url/id）")
    with io.open(path, encoding="utf-8") as f:
        models = json.load(f)
    for m in models:
        if m["id"] == model_key:
            return m["url"], m["apiKey"], m["id"]
    raise SystemExit(f"{path} 找不到模型 {model_key}")


def chat(url, key, model, messages, temperature=0.7, max_tokens=2000):
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.load(resp)
    usage = data.get("usage", {})
    msg = data["choices"][0]["message"]
    # 推理模型：content 可能为空而 reasoning_content 有货
    content = msg.get("content") or ""
    if not content.strip():
        content = msg.get("reasoning_content") or ""
    return content, usage


def parse_ops(text, nums):
    """从模型输出解析合法操作（两操作数须来自 nums，结果须算式正确）。"""
    ops = []
    pool = list(nums)
    for line in text.splitlines():
        m = OP_RE.match(line.strip().strip("`"))
        if not m:
            continue
        a, op, b, r = m.group(1), m.group(2), m.group(3), m.group(4)
        fa, fb, fr = Fraction(a), Fraction(b), Fraction(r)
        if fa not in pool or fb not in pool:
            continue
        # 算式校验
        try:
            calc = {"+": fa + fb, "-": fa - fb, "*": fa * fb,
                    "/": fa / fb if fb != 0 else None}[op]
        except ZeroDivisionError:
            continue
        if calc is None or calc != fr:
            continue
        ops.append((fa, op, fb, fr))
    return ops


SYS = ("你是 24 点游戏求解助手。给定一组数字，每次只能选其中两个数做 + - * / 运算，"
       "用掉这两个数、把结果放回数字集合，直到集合中剩下 24。"
       "每行输出一个候选操作，格式严格为：a op b = r（如 8 * 3 = 24），不要输出解释。"
       "优先给出你认为最有希望的多个不同候选（覆盖不同分支）。")


def state_nums(path):
    """path 是操作序列，返回剩余数字集合。"""
    pool = None
    for a, op, b, r in path:
        if pool is None:
            raise ValueError("path 必须从初始集合开始")
        pool = [x for x in pool if x not in (a, b)]
        pool.append(r)
    return pool


def step_candidates(url, key, model, nums, n, temperature, tracker):
    """一次调用生成候选操作，返回 [(a,op,b,r), ...]（已去重、保持顺序）。"""
    prompt = (f"当前数字集合：{[str(x) for x in nums]}\n"
              f"请给出最多 {n} 个不同的候选操作（每行一个，格式 a op b = r）：")
    messages = [{"role": "system", "content": SYS},
                {"role": "user", "content": prompt}]
    try:
        content, usage = chat(url, key, model, messages, temperature)
    except Exception as e:  # API 失败如实记账
        tracker["errors"].append(str(e)[:200])
        return []
    tracker["calls"] += 1
    tracker["in_tok"] += usage.get("prompt_tokens", 0)
    tracker["out_tok"] += usage.get("completion_tokens", 0)
    seen, out = set(), []
    for op in parse_ops(content, nums):
        k = (op[0], op[1], op[2])
        if k not in seen:
            seen.add(k)
            out.append(op)
    return out


def score_op(op):
    """与仿真一致的启发式（整数结果+温和量级+运算优先级）。"""
    r, o = op[3], op[1]
    is_int = r.denominator == 1
    in_range = 1 <= abs(r) <= 100
    op_rank = {"+": 3, "*": 2, "-": 1, "/": 0}[o]
    return (int(is_int), int(in_range), op_rank)


def width_v4(cands):
    """CEB v4 语义：候选分散度自适应——并列最优候选数=熵，决定保留宽度。"""
    if not cands:
        return 0
    top = max(score_op(c) for c in cands)
    ties = sum(1 for c in cands if score_op(c) == top)
    return {1: 3, 2: 2}.get(ties, 1)  # 1 并列=收敛(3宽探?) 见 POLICY 件


def width_v4_strict(cands):
    """v4 原语义（与 simharness.width_ceb_v4 逐字一致）：
    熵=最高分候选的并列质量数；width = max(1, min(n_tied, 4))。"""
    if not cands:
        return 1
    top = max(score_op(c) for c in cands)
    n_tied = sum(1 for c in cands if score_op(c) == top)
    return max(1, min(n_tied, 4))


WIDTHS = {"uniform-2": lambda c: 2, "uniform-3": lambda c: 3,
          "uniform-4": lambda c: 4, "ceb-v4": width_v4_strict}


def solve(nums, target, policy, url, key, model, tracker, temp,
          max_nodes=60, greedy=False):
    """beam search。返回 (solved, nodes, steps, path)。"""
    beam = [()]
    nodes = 0
    while beam and nodes < max_nodes:
        pool_of = {}
        nxt = []
        for path in beam:
            pool = state_nums(path) if path else list(nums)
            pool_of[id(path)] = pool
            if target in pool:
                return True, nodes, len(path), path
        cands_all = {}
        for path in beam:
            pool = pool_of[id(path)]
            cands = step_candidates(url, key, model, pool, 6, temp, tracker)
            cands_all[id(path)] = cands
        if not any(cands_all.values()):
            break
        for path in beam:
            pool = pool_of[id(path)]
            cands = sorted(cands_all[id(path)], key=score_op, reverse=True)
            w = 1 if greedy else WIDTHS[policy](cands)
            for op in cands[:w]:
                newpool = [x for x in pool if x not in (op[0], op[1])] + [op[3]]
                nodes += 1
                if target in newpool:
                    return True, nodes, len(path) + 1, path + (op,)
                nxt.append(path + (op,))
        # 去重 + 按路径分数截留（保持 beam 有界）
        seen, uniq = set(), []
        for p in nxt:
            k = tuple(state_nums(p))
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        beam = uniq[:8] if not greedy else uniq[:1]
    return False, nodes, -1, None


def probe(url, key, model):
    t = {"calls": 0, "in_tok": 0, "out_tok": 0, "errors": []}
    nums = [Fraction(x) for x in [3, 3, 8, 8]]
    prompt = (f"当前数字集合：{[str(x) for x in nums]}\n"
              f"请给出最多 6 个不同的候选操作（每行一个，格式 a op b = r）：")
    content, usage = chat(url, key, model,
                          [{"role": "system", "content": SYS},
                           {"role": "user", "content": prompt}], 0.7)
    print("=== 模型原始返回 ===")
    print(content[:800])
    print("=== 解析结果 ===")
    ops = parse_ops(content, nums)
    print(f"ops={[(str(o[0]),o[1],str(o[2]),'=',str(o[3])) for o in ops]}")
    return bool(ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--max-calls", type=int, default=500)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--models-json", default=None,
                    help="OpenAI 兼容 models 配置路径（含 apiKey/url/id）；"
                         "也可用环境变量 CEB_MODELS_JSON")
    ap.add_argument("--policies", default="uniform-3,uniform-4,ceb-v4")
    ap.add_argument("--out", default="llmverify.json")
    args = ap.parse_args()

    url, key, model = load_api(args.model, args.models_json)
    if args.probe:
        ok = probe(url, key, model)
        print("PROBE", "OK" if ok else "FAIL")
        return

    sys.path.insert(0, HERE)
    import ceb_simharness as H
    insts = H.gen_instances(args.seed, args.instances)
    policies = args.policies.split(",")

    tracker = {"calls": 0, "in_tok": 0, "out_tok": 0, "errors": []}
    results = {p: {"solved": 0, "nodes": 0, "steps": 0, "n": 0, "runs": []}
               for p in policies}
    greedy = {"solved": 0, "steps": 0, "n": 0}

    t0 = time.time()
    for i, inst in enumerate(insts):
        if inst.target is None:
            continue
        if tracker["calls"] > args.max_calls:
            print(f"[预算闸] 调用数 {tracker['calls']} 超 {args.max_calls}，停于实例 {i}")
            break
        nums = [Fraction(x) for x in inst.nums]
        # greedy 基线（pass@1 单链）
        gs, gn, gst, _ = solve(nums, inst.target, "uniform-4", url, key, model,
                               tracker, 0.0, greedy=True)
        greedy["n"] += 1
        greedy["solved"] += int(gs)
        greedy["steps"] += max(0, gst)
        for p in policies:
            if tracker["calls"] > args.max_calls:
                print(f"[预算闸] {p} 于实例 {i} 中止")
                results[p]["runs"].append({"inst": i, "aborted": True})
                break
            s, n, st, _ = solve(nums, inst.target, p, url, key, model,
                                tracker, 0.7)
            results[p]["n"] += 1
            results[p]["solved"] += int(s)
            results[p]["nodes"] += n
            results[p]["steps"] += max(0, st)
            results[p]["runs"].append({"inst": i, "solved": s, "nodes": n,
                                       "steps": st})
        print(f"  实例{i}: {[str(x) for x in inst.nums]} → "
              + " ".join(f"{p}={results[p]['runs'][-1].get('solved')}"
                         for p in policies if results[p]["runs"])
              + f" greedy={int(gs)} | 累计调用 {tracker['calls']}")

    out = {
        "model": model, "task": "24点", "instances": args.instances,
        "seed": args.seed, "policies": {},
        "greedy_pass1": {
            "solved": greedy["solved"], "n": greedy["n"],
            "rate": round(greedy["solved"] / max(1, greedy["n"]), 4),
            "avg_steps": round(greedy["steps"] / max(1, greedy["n"]), 2),
        },
        "cost": {k: tracker[k] for k in ["calls", "in_tok", "out_tok"]},
        "errors": tracker["errors"][:5],
        "wall_sec": round(time.time() - t0, 1),
        "budget_cap": args.max_calls,
    }
    for p in policies:
        r = results[p]
        n = max(1, r["n"])
        out["policies"][p] = {
            "solved": r["solved"], "n": r["n"],
            "rate": round(r["solved"] / n, 4),
            "avg_nodes": round(r["nodes"] / n, 1),
            "avg_steps": round(r["steps"] / max(1, r["solved"]), 2),
        }
    with io.open(os.path.join(HERE, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n→ {args.out} 已落盘")


if __name__ == "__main__":
    main()
