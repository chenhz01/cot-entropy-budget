# cot-entropy-budget (CEB) · 思考链熵预算

**step 级熵预算分配器** —— 生成前先规划整条推理链的熵分配，模型无关、可复算、带总预算约束。

## 它是什么（费曼层）

推理像走迷宫：有的路口要大胆多试几条（**分支期，高熵**），有的路口必须一锤定音（**收敛期，低熵**）。
现有工具要么逐 token 看模型"脸色"现场调（entropix），要么在训练时调熵（PRIME-RL）——
**没有一个在生成前先给整条推理链做"熵预算排期"的**。CEB 就是补这一层：先排期，后生成。

## 与相邻项目的精确差异（Step 2.5 五层扫描证据，2026-10-19）

| 项目 | 层级 | 机制 | CEB 增量 |
|---|---|---|---|
| entropix (3430★, 2024-11 停更) | 推理栈·token 级 | 逐 token 读 live logits（熵/方差熵）现场调采样，需模型权重 | CEB 是 step 级**预算规划**层，模型无关；entropix 无预算概念 |
| PRIME-RL/Entropy-Mechanism-of-RL (454★) | RL 训练级 | 熵正则（KL_Cov/Clip_Cov 已并入 verl） | CEB 服务推理时，不碰训练 |
| tree-of-thought-llm (6078★) | 搜索策略 | 固定广度树搜索 | CEB 广度按 step 自适应分配（分支高/收敛低） |

## 四件套

| 文件 | 角色 | 职责 |
|---|---|---|
| `ceb_allocator.py` | 主干 | 受总预算约束的 step 级熵档日程表 |
| `ceb_policy.py` | 卫星 | 分支/收敛检测政策（状态特征 > 文本启发） |
| `ceb_simharness.py` | 卫星 | 24 题真实实例 × 搜索推理体，uniform vs CEB 对比 |
| `ceb_realdata.py` | 卫星 | 真实推理链文本（任意语料，`--files` 传入）上的熵档分布跑批 |
| `ceb_publish.py` | 卫星 | OpenCore 发布脚手架（A藏配方/B送服务） |
| `ceb_llmverify.py` | 卫星 | 真模型复跑：LLM 候选分支 + 政策宽度，对比 pass@1/步数/成本 |

## 复现

```bash
PY=python3
cd cot-entropy-budget
$PY tests/smoke_ceb.py                                  # 5 项断言
$PY ceb_allocator.py --demo                              # 熵预算日程表演示
$PY ceb_simharness.py --instances 30 --seed 20261019 --cap 300
$PY ceb_realdata.py                                      # 真语料分布（包内 examples/）
$PY ceb_publish.py                                       # 生成本地发布物（不推送）
```

## 诚实边界（必读）

- 仿真 harness（24 点）测的是搜索推理体在真实可验证任务上的算力效率；
  文本档位判定是结构信号启发 `[推算]`，不是模型不确定性。
- 2026-10-19 独立复核（fresh-eyes）发现：v4 对 uniform-4 在 10 种子×30 实例
  pooled 下**系统性落后**（McNemar p<0.001），"效率换功率"结论不成立；
  v1.0 证伪记录为不可复现幽灵证据，已撤下。当前状态=**修订中**。
- 真模型复跑清单（`ceb_llmverify.py` / `ceb_llmverify_v2.py`）：DeepSeek flash
  跑 24 点，对比 greedy/uniform/CEB 的 pass@1、平均步数与调用成本；样本受预算闸限制，
  单一模型单任务，不外推为普适结论。**两个验证器均无中间落盘**（结果只在结尾写
  JSON）——长跑时进度不可观测，这是本次工程教训，使用者可自行加 `--progress`
  落盘改造。
