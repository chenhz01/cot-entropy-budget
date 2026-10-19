"""CEB-SIMHARNESS · 仿真推理体 × 真实可验证任务实例（卫星件 3 · Step 2.75 硬门槛）

为什么用 24 题：Game-of-24 是 Tree-of-Thoughts（princeton-nlp 6078★）同款
真实推理任务——4 个数字用 + - * / 各一次凑出 24，答案可机器验证。
ToT 用固定广度树搜索；本 harness 测 CEB 的 step 级熵预算日程：
把探索宽度按"分支期高 / 收敛期低"分配，是否比 uniform 更省算力、同等或更高解出率。

v1.1 政策迭代记录（Step 2.75 实证驱动，非拍脑袋）：
  v1.0（n_open>=3 全 HIGH）：首跑同解出率但节点 +176% → 宽度撒错地方，证伪。
  v2（root-heavy）：把熵预算集中到根节点（选哪对数=最高杠杆决策），
       深状态收敛决断。假设：根部分支的熵期望收益 > 深状态。
  最终政策由 sweep 数据定，不由偏好定。

诚实边界（[未验证]）：
  本机无 LLM API，测的是搜索推理体，不是 LLM CoT。
  "CEB 提升 LLM 推理"的真实增益需真模型复跑（见 README 复现清单）。

用法：
  python ceb_simharness.py --sweep                       # 全政策对比表
  python ceb_simharness.py --policy ceb-v2 --cap 10000   # 单政策
输出：ceb_simharness.out.json
"""
import argparse
import json
from fractions import Fraction
from itertools import combinations
import random

# ---------- 真实任务实例：Game of 24 ----------


def solvable(nums) -> bool:
    """穷举判定是否可解（精确有理数运算，无浮点误差）。"""
    def rec(state):
        if len(state) == 1:
            return state[0] == 24
        for i, j in combinations(range(len(state)), 2):
            a, b = state[i], state[j]
            rest = [state[k] for k in range(len(state)) if k not in (i, j)]
            for r in (a + b, a * b, a - b, b - a):
                if rec(tuple(rest + [r])):
                    return True
            if b != 0 and rec(tuple(rest + [Fraction(a, b)])):
                return True
            if a != 0 and rec(tuple(rest + [Fraction(b, a)])):
                return True
        return False
    return rec(tuple(Fraction(x) for x in nums))


def gen_instances(n: int, seed: int, lo: int = 1, hi: int = 13):
    """固定种子程序化生成 n 个可解实例（真实、可复算、可验证）。"""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        nums = tuple(sorted(rng.randint(lo, hi) for _ in range(4)))
        if nums in seen:
            continue
        seen.add(nums)
        if solvable(nums):
            out.append(nums)
    return out


# ---------- 宽度政策（sweep 对象） ----------

def _near_goal(state) -> bool:
    if len(state) != 2:
        return False
    a, b = state
    return (a + b == 24) or (a * b == 24) or (a - b == 24) or (b - a == 24) \
        or (b != 0 and Fraction(a, b) == 24) or (a != 0 and Fraction(b, a) == 24)


def _mk_uniform(k):
    return lambda state: k


def width_ceb_v1(state):
    """v1.0（已证伪）：>=3 未决全 HIGH(3)，2 未决 LOW(1)。"""
    return 3 if len(state) >= 3 else 1


def width_ceb_v2(state):
    """v2（root-heavy）：根(4 数) 宽度 4；3 数态宽度 2；收敛态宽度 1。"""
    n = len(state)
    if n >= 4:
        return 4
    if n == 3:
        return 2
    return 1


def width_ceb_v3(state):
    """v3（根极重+近门决断）：根 6；3 数态 2；2 数态临门 1、否则 2。"""
    n = len(state)
    if n >= 4:
        return 6
    if n == 3:
        return 2
    return 1 if _near_goal(state) else 2


def width_ceb_v4(state):
    """v4（分散度自适应 · 熵预算本义）：熵=最高分候选的并列质量数。
    前列候选得分完全并列 ⇒ 真不确定 ⇒ 花预算（宽度=并列数）；
    头名唯一 ⇒ 局势已定 ⇒ 宽度 1 省预算。
    与 v1/v2/v3 的区别：不看"还剩几个数"这种位置代理，直接量候选集的不确定度。"""
    cands = _candidates(state)
    if not cands:
        return 1
    top = cands[0][0]
    n_tied = sum(1 for c in cands if c[0] == top)
    return max(1, min(n_tied, 4))


POLICIES = {
    "uniform-2": _mk_uniform(2),
    "uniform-3": _mk_uniform(3),
    "uniform-4": _mk_uniform(4),
    "ceb-v1": width_ceb_v1,
    "ceb-v2": width_ceb_v2,
    "ceb-v3": width_ceb_v3,
    "ceb-v4": width_ceb_v4,
}


# ---------- 宽度受限搜索 ----------

def _candidates(state):
    """全部候选动作（选一对数 + 一个运算符），按启发式确定性排序。"""
    cands = []
    for i, j in combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        rest = [state[k] for k in range(len(state)) if k not in (i, j)]
        for op, r in (("+", a + b), ("*", a * b), ("-", a - b), ("r-", b - a)):
            cands.append((_score(r, op), op, tuple(rest + [r])))
        if b != 0:
            cands.append((_score(a / b, "/"), "/", tuple(rest + [Fraction(a, b)])))
        if a != 0:
            cands.append((_score(b / a, "r/"), "r/", tuple(rest + [Fraction(b, a)])))
    cands.sort(key=lambda c: c[0], reverse=True)
    return [(op, st) for _, op, st in cands]


def _score(r: Fraction, op: str) -> tuple:
    """启发式（v1.2 加 24 邻近性）：整数 + 与 24 的因子关系 + 温和量级 + 运算优先级。
    确定性排序，无随机。所有政策共用同一排序——比的是宽度分配，不是排序质量。"""
    is_int = r.denominator == 1
    factor_bonus = 0
    if r != 0 and (24 % r == 0 or r % 24 == 0):
        factor_bonus = 1
    in_range = 1 <= abs(r) <= 100
    op_rank = {"+": 3, "*": 2, "-": 1, "r-": 1, "/": 0, "r/": 0}[op]
    return (int(is_int), factor_bonus, int(in_range), op_rank)


def search(nums, width_of_state, cap: int):
    """DFS 宽度受限搜索。返回 (solved, nodes_expanded)。"""
    counter = [0]

    def dfs(state):
        if len(state) == 1:
            return state[0] == 24
        if counter[0] >= cap:
            return False
        w = width_of_state(state)
        for op, nxt in _candidates(state)[:w]:
            counter[0] += 1
            if counter[0] > cap:
                return False
            if dfs(nxt):
                return True
        return False

    ok = dfs(tuple(Fraction(x) for x in nums))
    return ok, counter[0]


def run_policy(instances, width_fn, cap: int):
    rows = []
    for nums in instances:
        ok, nodes = search(nums, width_fn, cap)
        rows.append({"instance": list(nums), "solved": ok, "nodes": nodes})
    solved = sum(1 for r in rows if r["solved"])
    nodes = [r["nodes"] for r in rows]
    solved_nodes = [r["nodes"] for r in rows if r["solved"]]
    return {
        "solved": solved, "total": len(rows),
        "solve_rate": round(solved / len(rows), 4),
        "avg_nodes": round(sum(nodes) / len(nodes), 1),
        "avg_nodes_solved": round(sum(solved_nodes) / len(solved_nodes), 1) if solved else None,
        "max_nodes": max(nodes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--policy", choices=sorted(POLICIES))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="ceb_simharness.out.json")
    args = ap.parse_args()

    inst = gen_instances(args.instances, args.seed)
    result = {
        "meta": {"instances": len(inst), "seed": args.seed, "cap": args.cap,
                 "harness": "ceb-simharness v1.1",
                 "note": "搜索推理体×24题真实实例；非LLM CoT实测 [未验证:真模型复跑]"},
        "policies": {},
    }
    names = sorted(POLICIES) if args.sweep else [args.policy or "ceb-v2"]
    for name in names:
        result["policies"][name] = run_policy(inst, POLICIES[name], args.cap)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 人读对比表
    print(f"{'policy':<10} {'solved':>7} {'rate':>7} {'avg_nodes':>10} {'solved_nodes':>13}")
    for name, r in result["policies"].items():
        print(f"{name:<10} {r['solved']:>4}/{r['total']:<2} {r['solve_rate']:>7.2%} "
              f"{r['avg_nodes']:>10} {str(r['avg_nodes_solved']):>13}")
    print(f"[写出] {args.out}（cap={args.cap}, instances={len(inst)}, seed={args.seed}）")


if __name__ == "__main__":
    main()
