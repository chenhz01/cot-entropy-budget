"""CEB-ALLOC · step 级熵预算分配器（主干件 · 方向核心）

输入：推理 step 列表 + 总熵预算 B。
输出：step 级熵档日程表（schedule），且总宽度支出受 B 硬约束。

这是与 entropix / PRIME-RL 的本质差异（Step 2.5 五层扫描证据）：
  - entropix（3430★, 2024-11 停更）：推理栈内逐 token 读 live logits 现场调采样，
    无预算概念、反应式、需模型权重。README 自述 "Entropy Based Sampling and
    Parallel CoT Decoding"——相邻但非 step 级预算规划。
  - PRIME-RL/Entropy-Mechanism-of-RL（454★）：RL 训练级熵正则（KL_Cov/Clip_Cov
    已并入 verl），属训练层，不涉推理步规划。
  - CEB-ALLOC（本件）：生成前先规划整条推理链的熵分配，模型无关、可复算、
    带总预算约束——"预算"语义是 entropix 完全没有的一维。

用法：
  python ceb_allocator.py --demo            # 内置演示 trace，打印日程表
  python ceb_allocator.py --steps steps.json --budget 12
"""
import argparse
import json

from ceb_policy import Step, policy_tier, schedule_width, TIER_WIDTH, HIGH, MED, LOW, PRUNE, DOWNGRADE_CHAIN

DOWNGRADE_ORDER = [HIGH, MED, LOW]   # 超支时先砍最高档（保收敛步优先）
UPGRADE_ORDER = [MED, HIGH]          # 节余时优先补分支信号最强的步


def allocate(steps, budget: int):
    """把 steps 排成受 budget 约束的熵档日程表。

    超支（sum(width) > budget）：按 DOWNGRADE_CHAIN 逐级降档
      （HIGH->MED->LOW->PRUNE），直到支出 <= budget。
      硬预算下没有特权步：收敛步也可被剪枝（PRUNE），如实记账。
    节余（sum(width) < budget）：按分支信号强度逐级补档
      （PRUNE->LOW->MED->HIGH），每次 +1 宽度。
    返回 (schedule, accounting)。任何 budget>=0 均可 feasible（PRUNE 兜底）。
    """
    tiers, reasons = [], []
    for s in steps:
        tier, why = policy_tier(s)
        tiers.append(tier)
        reasons.append(why)

    widths = schedule_width(tiers)
    actions = []

    def spend():
        return sum(widths)

    # --- 超支降档：每轮砍当前最高档的一个步 ---
    guard = 0
    while spend() > budget and guard < 10000:
        guard += 1
        for want in (HIGH, MED, LOW):
            hit = next((i for i, t in enumerate(tiers) if t == want), None)
            if hit is not None:
                old = tiers[hit]
                tiers[hit] = DOWNGRADE_CHAIN[want]
                widths = schedule_width(tiers)
                actions.append(f"step{steps[hit].idx} {old}->{tiers[hit]} (超支降档)")
                break
        else:
            break  # 全部 PRUNE 仍超支（仅当 budget<0，调用方责任）

    # --- 节余补档：每轮补分支信号最强且未满档的步 ---
    guard = 0
    while spend() < budget and guard < 10000:
        guard += 1
        cand = [i for i, t in enumerate(tiers) if t != HIGH]
        if not cand:
            break
        i = max(cand, key=lambda k: _branch_signal(steps[k]))
        old = tiers[i]
        tiers[i] = {PRUNE: LOW, LOW: MED, MED: HIGH}[old]
        widths = schedule_width(tiers)
        actions.append(f"step{steps[i].idx} {old}->{tiers[i]} (节余补档)")

    schedule = [{
        "step": s.idx, "kind": s.kind, "tier": tiers[i],
        "width": widths[i], "why": reasons[i],
    } for i, s in enumerate(steps)]
    accounting = {
        "budget": budget, "spend": sum(widths),
        "tiers": {HIGH: tiers.count(HIGH), MED: tiers.count(MED),
                  LOW: tiers.count(LOW), PRUNE: tiers.count(PRUNE)},
        "actions": actions,
        "within_budget": sum(widths) <= budget,
    }
    return schedule, accounting


def _branch_signal(s: Step) -> int:
    from ceb_policy import _hits, BRANCH_MARKERS
    return _hits(s.text, BRANCH_MARKERS) + s.n_candidates


def _demo_steps():
    """一条真实形状的推理链：开题分支 -> 多源交叉 -> 防御 -> 收敛结论。"""
    return [
        Step(0, text="从灵魂陈述出发判方向：可能往判据姿态或调度走，候选路径A/路径B"),
        Step(1, text="多源交叉：视频文案 + GitHub 扫描 + 同构比对，或者补 CDP 实抓"),
        Step(2, text="攻击：异议四件套——若根因是判据写错则主张降级"),
        Step(3, text="因此结论：主干定为 step 级熵预算分配器，答案为 CEB-ALLOC"),
    ]


def main():
    ap = argparse.ArgumentParser(description="CEB step 级熵预算分配器")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--steps", help="steps JSON（[{idx,text,n_open,near_goal,n_candidates}]）")
    ap.add_argument("--budget", type=int, default=8)
    args = ap.parse_args()

    if args.steps:
        raw = json.load(open(args.steps, encoding="utf-8"))
        steps = [Step(**r) for r in raw]
    else:
        steps = _demo_steps()

    schedule, accounting = allocate(steps, args.budget)
    print(json.dumps({"schedule": schedule, "accounting": accounting},
                     ensure_ascii=False, indent=2))
    if not accounting["within_budget"]:
        raise SystemExit("预算超支：分配器未守住约束（不应发生）")


if __name__ == "__main__":
    main()
