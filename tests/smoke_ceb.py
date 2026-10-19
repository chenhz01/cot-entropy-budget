"""CEB 冒烟测试 —— 五件各跑最小断言（写完把判定取反验红）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ceb_policy import Step, policy_tier, HIGH, MED, LOW, PRUNE  # noqa: E402
from ceb_allocator import allocate  # noqa: E402
import ceb_simharness as sim  # noqa: E402
import ceb_realdata as rd  # noqa: E402


def test_policy_state_tiers():
    # 分支期（>=3 未决）必 HIGH；收敛前夜必 LOW
    t1, _ = policy_tier(Step(0, n_open=4))
    t2, _ = policy_tier(Step(1, n_open=2, near_goal=True))
    t3, _ = policy_tier(Step(2, n_open=1))
    assert (t1, t2, t3) == (HIGH, LOW, LOW), (t1, t2, t3)


def test_policy_text_tiers():
    t_hi, _ = policy_tier(Step(0, text="可能走路径A，或者备选方案，候选两个都待验"))
    t_lo, _ = policy_tier(Step(0, text="因此结论：答案为 CEB-ALLOC"))
    assert t_hi == HIGH and t_lo == LOW, (t_hi, t_lo)


def test_allocator_budget_hard_constraint():
    # budget=1、2 步：PRUNE 兜底后必须 feasible（v1.0 曾在此失败）
    steps = [Step(i, text="可能候选" * (i + 1)) for i in range(2)]
    _, acc = allocate(steps, budget=1)
    assert acc["within_budget"] and acc["spend"] <= 1, acc
    assert acc["tiers"][PRUNE] >= 1, acc
    # budget=0：全部 PRUNE，支出 0
    _, acc0 = allocate(steps, budget=0)
    assert acc0["spend"] == 0 and acc0["within_budget"], acc0
    # 节余：budget 大于自然支出时补档到 HIGH
    _, acc_big = allocate([Step(0, text="因此结论")], budget=9)
    assert acc_big["tiers"][HIGH] == 1, acc_big


def test_simharness_real():
    inst = sim.gen_instances(5, seed=20261019)
    assert len(inst) == 5 and all(sim.solvable(x) for x in inst)
    r = sim.run_policy(inst, sim.POLICIES["ceb-v2"], cap=10000)
    assert r["total"] == 5 and r["solved"] >= 1, r


def test_realdata_parse():
    out = rd.run()
    assert out["meta"]["total_steps"] > 0, "真语料零命中：路径或解析失效"
    # 表头脏行已过滤：样本不应出现"推演链（输入"这类列名碎片
    bad = [s for s in out["samples"] if s["text"].startswith(("推演链", "动作", "输出"))]
    assert not bad, f"表头脏行未过滤: {bad}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"[ALL PASS] {len(fns)} 项")
