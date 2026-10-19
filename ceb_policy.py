"""CEB-POLICY · 思考链熵预算 · 分支/收敛检测政策（卫星件 2）

纯 stdlib。职责：把「一步推理」的特征映射到离散熵档（HIGH/MED/LOW），
是主干分配器 ceb_allocator.py 的上游。

与 entropix（3430★）的本质差异（Step 2.5 扫描结论）：
  entropix = 推理栈内逐 token 读 live logits（熵/方差熵）现场调采样，需模型权重；
  本政策 = 只看 step 级结构特征（搜索状态或文本信号），输出离散熵档，
            可事前规划、可复算、模型无关（不持有任何模型）。

文本信号为启发式代理 [推算]：非 LLM 不确定性，只是结构信号词命中，
真实 LLM 推理的熵需 logprobs，本机无模型 API，已诚实标注边界。
"""
from dataclasses import dataclass, field

HIGH, MED, LOW = "HIGH", "MED", "LOW"
PRUNE = "PRUNE"   # 硬预算下的剪枝档：该步不分配探索宽度（width=0）
# 熵档 -> 分支宽度（该步最多展开的候选续写数）
TIER_WIDTH = {HIGH: 3, MED: 2, LOW: 1, PRUNE: 0}
# 降档链（超支时逐级下放）：HIGH -> MED -> LOW -> PRUNE
DOWNGRADE_CHAIN = {HIGH: MED, MED: LOW, LOW: PRUNE}

# --- 文本信号词表（启发式，可被独立复核后调整） ---
BRANCH_MARKERS = ["可能", "也许", "要么", "或者", "如果", "假设", "待验",
                  "候选", "备选", "交叉", "多源", "？", "?", "路径A", "方案A"]
CONVERGE_MARKERS = ["因此", "所以", "结论", "答案", "综上", "定为", "故", "即"]
FALSIFY_MARKERS = ["证伪", "翻掉", "反例", "推翻", "边界", "什么证据说明"]


@dataclass
class Step:
    """一步推理。kind 可由状态特征或文本自动判定。"""
    idx: int
    kind: str = "unknown"          # branch / converge / verify / unknown
    text: str = ""
    n_open: int = 0                # 未决要素数（搜索态：剩余数字数）
    near_goal: bool = False        # 一步可达目标（收敛前夜）
    n_candidates: int = 0          # 候选续写数（>1 即分支信号）

    def as_features(self) -> dict:
        return {
            "idx": self.idx, "kind": self.kind, "n_open": self.n_open,
            "near_goal": self.near_goal, "n_candidates": self.n_candidates,
            "branch_hits": _hits(self.text, BRANCH_MARKERS),
            "converge_hits": _hits(self.text, CONVERGE_MARKERS),
            "falsify_hits": _hits(self.text, FALSIFY_MARKERS),
        }


def _hits(text: str, markers: list) -> int:
    if not text:
        return 0
    return sum(1 for m in markers if m in text)


def classify_state(n_open: int, near_goal: bool) -> str:
    """搜索推理体的状态判定：未决要素多=分支期，临门=收敛期。"""
    if n_open >= 3:
        return "branch"
    if n_open == 2:
        return "converge" if near_goal else "branch"
    return "converge"


def classify_text(text: str) -> str:
    """文本判定（启发式代理 [推算]）。分支信号压过收敛信号即分支。"""
    b, c = _hits(text, BRANCH_MARKERS), _hits(text, CONVERGE_MARKERS)
    if b > c:
        return "branch"
    if c > 0:
        return "converge"
    return "unknown"


def policy_tier(step: Step):
    """返回 (tier, reasons)。判定优先级：状态特征 > 文本启发。"""
    reasons = []
    if step.n_open or step.near_goal:
        kind = classify_state(step.n_open, step.near_goal)
        step.kind = kind
        if step.n_open >= 3:
            reasons.append(f"未决要素 {step.n_open} 个>=3 -> 分支期")
        elif step.n_open == 2 and step.near_goal:
            reasons.append("剩余 2 要素且一步可达 -> 收敛前夜")
        elif step.n_open == 2:
            reasons.append("剩余 2 要素未临门 -> 分支期")
        else:
            reasons.append("仅剩 1 要素 -> 收敛")
        tier = {"branch": HIGH, "converge": LOW}[kind]
        return tier, reasons

    b, c = _hits(step.text, BRANCH_MARKERS), _hits(step.text, CONVERGE_MARKERS)
    f = _hits(step.text, FALSIFY_MARKERS)
    score = b - c
    if score >= 2:
        tier, why = HIGH, f"分支信号 {b} 命中压过收敛 {c}"
    elif score == 1:
        tier, why = MED, f"分支信号 {b} 略占优"
    else:
        tier, why = LOW, f"收敛/证伪信号主导 (b={b}, c={c}, f={f})"
    # [v1.1 修复 G5] f 参与判档：证伪/反例命中是「主动探索不确定性」的信号，
    # 至少升一档（LOW->MED, MED->HIGH）。修复前 f 只进 reason 文案、从不判档
    # （死代码）——fresh-eyes 复核员实证：13/14 证伪步被判 LOW。
    if f >= 2 or (f >= 1 and score >= 1):
        upgraded = {LOW: MED, MED: HIGH, HIGH: HIGH}[tier]
        if upgraded != tier:
            why += f"；证伪命中 {f} -> {tier}->{upgraded}（f 判档修复 v1.1）"
            tier = upgraded
    step.kind = classify_text(step.text)
    reasons.append(why + " [推算:文本启发]")
    return tier, reasons


def schedule_width(tiers: list) -> list:
    return [TIER_WIDTH[t] for t in tiers]
