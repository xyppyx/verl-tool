# CARE-Tool Research Plan

> 暂定全称：**Counterfactual Attribution and Reliability-aware Exploration for Tool Agents**  
> 中文：面向不可靠工具环境的反事实信用分配与可靠性感知探索  
> 文档定位：在 `VERLTOOL_ENGINEERING_PROJECT.md` 已完成的工程基线上，升级为论文型研究项目  
> 最后核验日期：2026-07-16  
> 重要声明：本文描述的是研究假设、技术方案和实验协议，不代表已经获得结果，也不保证达到任何会议录用标准。

---

## 0. 前置条件与 Codex 执行边界

开始本文件前，必须满足：

- `VERLTOOL_ENGINEERING_PROJECT.md` 至少达到 E1；
- clean baseline 可重复运行；
- fault benchmark 已实现；
- injected external fault 与 agent-caused error 可区分；
- Static-Fault baseline 已有结果；
- trajectory 中可定位 fault observation 的 token 边界；
- 当前实验配置、seed 和上游 commit 已冻结。

若以上条件不满足，不得直接改 advantage estimator。

### 0.1 论文项目成功的三档定义

| 档位 | 条件 | 建议定位 |
|---|---|---|
| R0：研究信号 | paired rollout 或 local credit 在一类故障上有稳定趋势 | 找老师、内部技术报告 |
| R1：方法成立 | 核心消融成立，clean 不退化，多故障和多 seed 有效 | workshop / 短文候选 |
| R2：完整论文 | 两个环境、强基线、OOD 泛化、资源公平、理论或估计器分析 | 主会尝试 |

---

# 1. STAR 法则：论文背景与研究目标

## S — Situation：研究背景

Agentic RL 将语言模型置于多轮环境交互中。与单轮 reasoning RL 相比，Agent trajectory 包含模型动作、工具 observation、状态转移和随机环境因素：

\[
\tau=(x,a_1,o_1,a_2,o_2,\ldots,a_T,y)
\]

现有 VerlTool 已经能统一运行工具交互与 GRPO/DAPO，但典型训练仍以最终任务奖励为主。普通序列级 GRPO 对同一条 trajectory 的所有有效模型 token 广播同一 advantage；observation token 虽被 mask，但不同模型动作仍可能共享同一终局信用。

工具环境存在两类失败：

1. **内生错误（endogenous error）**：模型选择错误工具、参数错误、代码语法错误或状态判断错误；
2. **外生故障（exogenous fault）**：模型动作正确，但服务 timeout、临时不可用、返回为空或 observation 截断。

考虑轨迹：

```text
正确推理
→ 正确工具调用
→ 外部服务 timeout
→ 模型错误地提前结束
→ 最终失败
```

标准 trajectory-level 负奖励可能同时抑制：

- 故障前正确 reasoning；
- 正确工具调用；
- 故障后错误恢复。

这构成 **external-fault credit contamination**：环境随机故障污染了模型动作的信用估计。

该方向与现有工作相邻但不相同：

- Fission-GRPO 将模型产生的执行错误转化为带诊断反馈的 on-policy 恢复训练实例；
- 通用 Agentic RL credit assignment 研究关注 token、step、turn 或 trajectory 的细粒度归因；
- CRAFT 等近期方法利用 sibling rollout 估计反事实 token credit；
- 本项目聚焦“同一正确工具动作在 clean/fault 环境转移下的因果隔离”，并只更新故障后的恢复策略。

因此，论文不得声称“首次研究工具恢复”“首次研究反事实 credit assignment”或“首次对 Agent 做局部信用”。可主张的窄问题是：

> 如何在可控外生工具故障下，通过共享动作前缀的 clean/fault 配对分支，避免故障结果错误归因给故障前模型动作，并提高故障后的恢复策略学习效率？

## T — Task：研究任务

研究目标包括：

1. 量化外生工具故障对序列级 GRPO 的信用污染；
2. 构造严格共享 prefix 的 clean/fault counterfactual pair；
3. 用 clean 分支估计故障前 action prefix 的质量；
4. 只对 fault observation 后的模型 token 分配恢复信用；
5. 在相同 rollout/token 预算下与独立故障采样、Static-Fault 和恢复训练基线比较；
6. 通过学习进展课程提高低预算样本效率；
7. 在隐藏动态工具可靠性的多工具环境中验证工具路由和恢复；
8. 评测平均性能、worst-group、OOD fault 和 clean/fault robustness gap。

核心假设：

- H1：外生故障会降低故障前正确动作的优化信号质量；
- H2：共享 prefix 的 clean/fault pair 比独立采样具有更低的比较噪声；
- H3：fault suffix-only credit 能提高条件恢复率，并减少 clean 性能退化；
- H4：学习进展课程可在固定 rollout 预算下提高恢复学习效率；
- H5：在动态可靠性环境中，模型能学会根据历史 observation 重试或切换工具。

## A — Action：融合方法

方法分四层，有明确依赖关系：

1. **Paired Counterfactual Rollout**：同一 action prefix 在 clean/fault observation 下分叉；
2. **Post-Fault Local Credit Assignment**：共享 prefix 不受 fault outcome 污染，fault suffix 接收恢复优势；
3. **Learning-Progress Reliability Curriculum**：按不同 fault 的恢复率和学习进展动态采样；
4. **Hidden Dynamic Reliability + Risk Objective**：多工具的可靠性、成本和延迟隐藏且可漂移，使用 worst-group/CVaR 评测或训练。

其中 1+2 是核心方法；3 是低算力增强；4 是完整论文环境与鲁棒性扩展。

## R — Result：预期证据链

论文成立需要同时得到：

1. 普通 Static-Fault 训练存在 prefix contamination 或 clean 性能退化证据；
2. Paired rollout 优于 independent clean/fault sampling；
3. Paired + Local Credit 优于 Paired-only；
4. 提升不能由更多 rollout、更多 fault 数据或更长上下文解释；
5. clean accuracy 基本保持；
6. recovery rate、worst-group 和 OOD fault 至少有两项稳定改善；
7. 多 seed 趋势一致；
8. 至少第二个环境或第二种工具任务复现主要结论；
9. 与 Fission-GRPO 类恢复训练和一个通用细粒度 credit baseline 有清晰定位或实证比较。

预期论文叙事：

> Existing trajectory-level Agentic RL conflates policy errors with stochastic external tool failures. CARE-Tool constructs paired clean/fault sibling trajectories sharing an identical action prefix, uses the clean branch to preserve credit for pre-fault decisions, and optimizes only post-fault recovery actions with localized advantages. A learning-progress curriculum improves sample efficiency, while hidden dynamic tool reliability evaluates adaptive retry and routing under risk.

---

# 2. 研究 gap 的边界与相关工作定位

## 2.1 已被覆盖的方向

以下内容不能作为独立 novelty claim：

- 给工具注入错误；
- 训练 Agent 处理 timeout；
- 使用课程学习逐渐增加故障；
- 从失败轨迹生成恢复样本；
- 对 Agent trajectory 做 step/turn-level credit；
- 使用 counterfactual sibling rollout；
- 使用 CVaR 或 worst-group 优化。

## 2.2 本项目的可守住差异

| 方向 | 主要对象 | 信号来源 | CARE-Tool 的区别 |
|---|---|---|---|
| Static Fault Training | 故障数据增强 | 终局奖励 | 不解决 prefix 污染 |
| Fission-GRPO | 模型执行错误后的纠错 | Error Simulator + recovery resampling | CARE 聚焦外生故障，并以 clean sibling 隔离故障前动作质量 |
| 通用 step/turn credit | 一般长轨迹动作 | PRM、critic、hindsight 等 | CARE 利用可控环境 bifurcation，不需要为所有 step 建模 |
| CRAFT/Sibling credit | sibling rollout 的 token credit | group counterfactual estimate | CARE 的 sibling 差异由同一工具动作后的环境 intervention 产生，目标是 post-fault recovery |

## 2.3 必须谨慎的 claim

推荐：

> We study credit contamination caused specifically by exogenous tool failures and introduce controlled clean/fault bifurcation with post-fault localized policy updates.

不推荐：

> We solve credit assignment for Agentic RL.

---

# 3. 形式化问题定义

## 3.1 环境

令状态为 `s_t`，模型动作为 `a_t`，工具 observation 为 `o_t`。环境含隐藏故障变量：

\[
z_t \in \{\text{clean}, \text{timeout}, \text{empty}, \text{truncated}, \ldots\}
\]

转移：

\[
o_t \sim P(o_t\mid s_t,a_t,z_t)
\]

外生故障要求：在 `a_t` 合法且语义正确时，由 `z_t` 改变 observation，而不是修改模型 action。

## 3.2 错误类型

```text
agent_error:
  invalid syntax / invalid parameter / wrong tool / wrong state action

environment_fault:
  timeout / transient unavailable / empty / truncation
```

只有 `environment_fault` 进入 CARE 的 counterfactual credit 路径。

## 3.3 普通 GRPO 的问题

对同一 prompt 的 `G` 条 trajectory：

\[
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}
\]

普通序列级方法将 `A_i` 广播给所有模型 token：

\[
A_{i,t}=A_i
\]

若最终失败由外生 `z_k` 导致，则 `t \le k` 的正确模型动作也被负向更新。

---

# 4. 核心方法一：共享前缀 clean/fault 配对 rollout

## 4.1 严格配对条件

在合法工具 action `a_k` 产生后，冻结：

\[
p=(x,a_1,o_1,\ldots,a_k)
\]

复制 trajectory 状态并生成两个分支：

\[
\tau^C=(p,o_k^C,a_{k+1}^C,\ldots)
\]

\[
\tau^F=(p,o_k^F,a_{k+1}^F,\ldots)
\]

严格要求：

- 相同 prompt；
- 相同模型 checkpoint；
- 相同故障前 token；
- 相同工具 action；
- 相同环境初始状态；
- 唯一 intervention 是 fault observation。

独立采样两个 trajectory 不满足严格配对。

## 4.2 环境状态复制

Tool Server 必须支持：

```python
snapshot = env.snapshot(trajectory_id)
clean_id = env.clone(snapshot)
fault_id = env.clone(snapshot)
```

对于无状态 Python calculator，可先只复制 metadata；对于 SQLite 或状态工具，必须实现真正快照，避免分支互相污染。

## 4.3 Pair metadata

每个分支保存：

```json
{
  "pair_id": "sample-42-prefix-1",
  "branch": "clean",
  "fault_type": null,
  "fault_turn": 1,
  "prefix_token_end": 318,
  "post_fault_model_mask": [0, 0, 1, 1],
  "environment_snapshot_id": "..."
}
```

fault 分支保存同一个 `pair_id`。

## 4.4 Rollout 预算控制

低成本配置可用：

```text
每 prompt 采样 2 个 action prefix
每个 prefix 分成 clean/fault
总 trajectory 数 = 4
```

与普通 `rollout_n=4` 保持相同 trajectory 数。需要额外报告 clean/fault suffix 的总 token，防止 pair 方法实际消耗更多计算。

---

# 5. 核心方法二：故障后局部信用分配

## 5.1 三段式 token 区域

每个 pair 分成：

1. shared prefix；
2. clean suffix；
3. fault suffix。

已有 `response_mask`：

\[
M_t^{response}=1 \text{ for model tokens},\quad 0 \text{ for observations}
\]

新增：

\[
M_t^{post}=1[t>t_{fault}]
\]

## 5.2 最低风险版本：Local-Mask CARE

clean 分支：

\[
M_t^C=M_t^{response}
\]

fault 分支：

\[
M_t^F=M_t^{response}M_t^{post}
\]

总体 loss：

\[
\mathcal L=\mathcal L_{clean\ full}+\lambda_F\mathcal L_{fault\ suffix}
\]

优点：

- 不需立即修改 GRPO advantage 公式；
- 能直接测试“fault 结果是否不应污染 prefix”；
- 容易做 bit-level mask 验证。

局限：

- fault 分支不对 prefix 提供任何信息；
- clean 分支的终局奖励仍是粗粒度的；
- 不是完整反事实 advantage estimator。

## 5.3 完整版本：Paired Recovery Advantage

定义 clean/fault 终局结果：

\[
C_i\in\{0,1\},\quad F_i\in\{0,1\}
\]

只有 clean 成功的 pair 才可用于恢复学习：

\[
V_i=C_i
\]

条件恢复目标：

\[
S_i^{rec}=C_iF_i
\]

在有效 pair 中标准化 fault outcome：

\[
A_i^{rec}=C_i\frac{F_i-\mu_F}{\sigma_F+\epsilon}
\]

建议 token advantage：

\[
A_{i,t}^{CARE}=
\begin{cases}
A_i^C, & \text{clean branch}\
A_i^F+\lambda_{rec}A_i^{rec}, & \text{fault branch and }t>t_{fault}\
0, & \text{fault branch and }t\le t_{fault}
\end{cases}
\]

注意：`A_i^C`、`A_i^F` 的 group 定义必须明确。可选方案：

- clean branch 在 clean sibling group 内标准化；
- fault branch 在同 fault type 的有效 pair 内标准化；
- 或使用 joint group，但必须做消融。

## 5.4 不直接使用 `F-C` 作为训练奖励

`F-C` 在 clean/fault 都成功时为 0，不能正向奖励成功恢复。建议：

- `C-F` 仅作为 robustness gap；
- `C·F` 或条件 fault outcome 作为 recovery 学习信号。

## 5.5 Credit contamination 诊断

为了证明问题真实存在，记录故障前 action 的 log probability 或梯度方向：

- Static-Fault 下，clean-success/fault-fail pair 的 prefix log-prob 变化；
- Local-Mask 下，同一 prefix 的变化；
- 统计合法工具 action probability 是否被错误压低。

可选指标：

\[
\Delta_{prefix}=\log\pi_{after}(a_{1:k}|x)-\log\pi_{before}(a_{1:k}|x)
\]

如果 Static-Fault 对正确 prefix 产生系统性负变化，而 CARE 缓解，则形成强动机证据。

---

# 6. 辅助方法：学习进展可靠性课程

## 6.1 统计量

对每种 fault `k`，维护条件恢复率 EMA：

\[
s_k^{(t)}=\beta s_k^{(t-1)}+(1-\beta)\hat s_k^{(t)}
\]

学习进展：

\[
LP_k^{(t)}=s_k^{(t)}-s_k^{(t-\Delta)}
\]

## 6.2 采样权重

\[
w_k=(1-s_k)\left(\max(LP_k,0)+\epsilon\right)
\]

加入最低探索：

\[
p_k'=(1-\eta)\frac{w_k}{\sum_jw_j}+\frac{\eta}{|\mathcal F|}
\]

## 6.3 目的

- 已掌握 fault：降低采样；
- 当前可学习 fault：提高采样；
- 极难且无进展 fault：保留探索但不浪费大部分预算。

## 6.4 必须公平比较

课程方法必须固定：

- 总 pair 数；
- 总生成 token；
- 每类 fault 最低曝光；
- 训练 step；
- 模型与 seed。

主张应是“样本效率提升”，而不是“用了更多困难数据”。

---

# 7. 完整环境：隐藏动态工具可靠性

该阶段只在核心方法已成立后实现。

## 7.1 多工具设计

| Tool | 能力 | 成本 | 延迟 | 初始可靠性 |
|---|---|---:|---:|---:|
| FastCalculator | 基础算术 | 1 | 1 | 隐藏，中等 |
| ReliableCalculator | 基础算术 | 2 | 2 | 隐藏，较高 |
| PythonInterpreter | 通用计算 | 4 | 4 | 隐藏，较高 |

每个 episode 采样可靠性：

\[
\theta_j\sim Beta(\alpha_j,\beta_j)
\]

Agent 不直接观察 `θ_j`，只能从历史 observation 推断。

## 7.2 可靠性漂移

可选：

\[
\theta_j^{t+1}=clip(\theta_j^t+\epsilon_t,0,1)
\]

或在随机 turn 发生 regime switch。测试 Agent 是否会：

- 首次失败后重试；
- 连续失败后切换工具；
- 简单任务选择低成本工具；
- 高风险任务选择可靠工具。

## 7.3 奖励

禁止只用无条件成本惩罚，否则模型可能学会不调用工具。建议：

\[
R=R_{success}(1-\lambda_c C-\lambda_t T)-\lambda_i N_{invalid}
\]

成本项主要在成功条件下生效。

## 7.4 程序生成任务

为降低标注成本，可生成：

- 多步算术；
- 查询 JSON 后计算；
- SQLite 查询后计算；
- 同一答案需要不同工具路径的任务。

所有任务必须有 deterministic verifier。

---

# 8. 风险敏感评测与可选优化

## 8.1 主要评测

\[
RobustnessGap=Acc_{clean}-Acc_{fault}
\]

\[
WorstGroup=\min_k Acc_k
\]

\[
RecoveryRate=P(F=1|C=1)
\]

## 8.2 CVaR

对 fault episode 回报的最差 `α` 分位：

\[
CVaR_\alpha(R)=E[R|R\le q_\alpha]
\]

第一版仅作为评测；核心方法稳定后再加入训练。

## 8.3 为什么不是主创新

CVaR 与 worst-group 是成熟鲁棒优化工具，单独套用到 VerlTool 不足以构成主要贡献。它们用于证明 CARE 不只提高平均 fault accuracy。

---

# 9. 实现规划

## 9.1 建议模块

```text
care_tool/
├── branching/
│   ├── pair_manager.py
│   ├── env_snapshot.py
│   ├── sibling_scheduler.py
│   └── metadata.py
├── credit/
│   ├── local_mask.py
│   ├── paired_advantage.py
│   ├── group_normalization.py
│   └── diagnostics.py
├── curriculum/
│   ├── progress_tracker.py
│   ├── sampler.py
│   └── state_io.py
├── environments/
│   ├── reliability_model.py
│   ├── multi_tool_math.py
│   └── sqlite_calc.py
├── metrics/
│   ├── recovery.py
│   ├── prefix_credit.py
│   ├── risk.py
│   └── efficiency.py
└── integration/
    ├── verltool_agent_loop_patch.py
    ├── trainer_patch.py
    └── reward_manager_patch.py
```

## 9.2 修改优先级

1. metadata，不改变训练；
2. clean/fault evaluation pair，不改变训练；
3. local mask；
4. paired advantage；
5. curriculum；
6. multi-tool environment；
7. risk objective。

每一步必须有独立配置开关，关闭后应尽量恢复 baseline 行为。

## 9.3 配置开关

```yaml
care:
  enabled: false
  paired_rollout: false
  local_suffix_mask: false
  paired_advantage: false
  curriculum: false
  dynamic_reliability: false
  risk_objective: none
```

禁止通过复制多份 trainer 形成不可维护分支。

## 9.4 Bit-exact 或近似等价检查

当 `care.enabled=false` 时：

- 相同 seed；
- 相同配置；
- 相同 batch；
- rollout 和 loss 应与 baseline 一致或明确记录不可避免的异步差异。

至少验证：

- mask；
- reward；
- advantage；
- loss；
- optimizer step。

---

# 10. 实验矩阵

## 10.1 核心消融

| ID | Method | Paired | Local Credit | Curriculum | Dynamic Reliability |
|---|---|---:|---:|---:|---:|
| B0 | Base Model | 0 | 0 | 0 | 0 |
| B1 | Clean GRPO | 0 | 0 | 0 | 0 |
| B2 | Static-Fault GRPO | 0 | 0 | 0 | 0 |
| B3 | Independent Clean/Fault | 0 | 0 | 0 | 0 |
| M1 | Paired Rollout | 1 | 0 | 0 | 0 |
| M2 | Paired + Local Mask | 1 | 1 | 0 | 0 |
| M3 | Paired + Paired Advantage | 1 | 1 | 0 | 0 |
| M4 | CARE-Core | 1 | 1 | 1 | 0 |
| M5 | CARE-Dynamic | 1 | 1 | 1 | 1 |

最关键比较：

```text
B3 vs M1：共享 prefix 是否重要
M1 vs M2：局部 credit 是否重要
M2 vs M3：完整 advantage 是否超过简单 mask
M3 vs M4：课程是否提高样本效率
```

## 10.2 环境

最低论文版本：

- Environment A：Math + Python；
- Environment B：程序生成的 SQLite/JSON + Calculator 多工具任务。

若只有 Math + Python，只适合作为 preliminary 或 workshop 风险较低版本。

## 10.3 模型

资源允许时：

- Qwen2.5-Math-1.5B；
- 第二模型：Qwen3 1.7B/4B 或另一同量级 instruct model。

不能只比较同一模型的不同 checkpoint 并宣称跨模型泛化。

## 10.4 Seed

- 开发：1 seed；
- 主表：至少 3 seeds；
- 若算力不足，核心方法主指标 3 seeds，其余消融 1–2 seeds，并明确说明。

## 10.5 故障分布

训练故障：

- timeout；
- transient error；
- empty observation；
- truncation level A。

OOD 测试：

- 更高故障率；
- truncation level B；
- rate limit；
- episode 内 reliability drift。

---

# 11. 指标与诊断

## 11.1 主指标

- Clean Accuracy；
- Fault Accuracy；
- Conditional Recovery Rate；
- Robustness Gap；
- Worst-group Accuracy；
- Average Tool Calls；
- Generated Tokens；
- GPU Hours。

## 11.2 Credit 指标

- correct pre-fault action probability；
- prefix log-prob change；
- fault suffix success probability；
- post-fault immediate-abort rate；
- repeated invalid retry rate；
- gradient norm by token region。

## 11.3 工具路由指标

- retry rate；
- switch rate；
- successful switch rate；
- cost per successful task；
- reliability adaptation after consecutive faults。

## 11.4 统计规则

- 报告均值与标准差或 bootstrap CI；
- 对 paired clean/fault 结果使用 paired statistical test；
- 多组比较时避免只挑显著结果；
- 报告所有预注册主指标。

---

# 12. 理论与分析方向

完整主会版本最好至少完成一项。

## 12.1 偏差分析

定义真实 prefix action quality：

\[
Q^{clean}(p)=E[R|p,z=clean]
\]

普通 fault trajectory 使用：

\[
E[R|p,z\sim P(z)]
\]

当外生故障概率与恢复能力导致终局回报下降时，后者不是纯粹的 prefix action quality 估计。clean sibling 提供受控估计：

\[
\hat Q^{clean}(p)=R(\tau^C)
\]

需要清楚说明该估计仍受 suffix sampling variance 影响。

## 12.2 方差分析

共享 prefix pair 相比独立 rollout 消除部分任务和 prefix 随机性。可实证测量：

- paired difference 方差；
- independent difference 方差；
- advantage variance；
- gradient variance。

## 12.3 假设限制

方法依赖：

- 故障可以被识别为外生；
- 可以执行 clean counterfactual；
- 环境可复制或无状态；
- clean 分支本身是合理参照；
- 工具故障不会永久改变不可恢复的真实世界状态。

对支付、删除等不可逆工具，不能无条件做 counterfactual replay。

---

# 13. 强基线与相关工作比较

至少包含：

1. Clean GRPO；
2. Static-Fault GRPO；
3. Independent clean/fault sampling；
4. Fission-GRPO 风格恢复训练或可复现近似；
5. 一个通用 local/turn-level credit baseline；
6. CARE local mask；
7. CARE paired advantage。

若无法完整复现外部方法：

- 明确写“adapted baseline”；
- 对齐模型、数据和预算；
- 不使用论文原始不同规模结果替代本地对比。

---

# 14. 论文推进 Go/No-Go 标准

## Gate A：问题是否存在

继续条件：Static-Fault 确实出现以下至少一项：

- clean 性能下降；
- 正确 prefix 概率被压低；
- fault recovery 低；
- 重复错误调用明显。

否则重新审视问题设定。

## Gate B：核心方法是否有效

继续条件：M2 或 M3 相比 B2/M1 在至少两个 fault type 上稳定改善 recovery，且 clean 无显著退化。

否则保持工程项目，不堆叠课程与多工具环境。

## Gate C：是否具有论文扩展性

继续条件：

- 第二环境复现；
- OOD fault 有泛化；
- 资源公平后仍有优势；
- 3 seed 趋势稳定。

## Gate D：是否尝试主会

建议满足：

- 核心方法和强基线完整；
- 两环境或两任务族；
- 相关工作定位清晰；
- 估计器或方差分析；
- 可复现代码与配置；
- 论文不是只靠一个超参数或 fault type。

---

# 15. 论文结构草案

## 1 Introduction

- Agentic RL 的多轮工具交互；
- 外生工具故障与模型错误混淆；
- trajectory-level credit contamination；
- clean/fault bifurcation；
- 贡献列表。

## 2 Related Work

- Agentic RL with tools；
- tool error recovery；
- credit assignment；
- counterfactual/sibling rollout；
- robust and risk-sensitive RL。

## 3 Problem Formulation

- MDP/POMDP；
- endogenous vs exogenous failure；
- paired intervention；
- metrics。

## 4 Method

- paired rollout；
- local mask；
- paired advantage；
- curriculum；
- optional dynamic reliability。

## 5 Experiments

- environments；
- models；
- baselines；
- budget；
- main results；
- ablations；
- OOD；
- efficiency。

## 6 Analysis

- prefix credit contamination；
- variance；
- trajectory cases；
- failure boundaries。

## 7 Limitations

- counterfactual execution cost；
- environment snapshot requirement；
- irreversible tools；
- fault attribution oracle；
- small-model limitations。

---

# 16. Codex 阶段任务

## R-M0：研究数据层

- [ ] pair metadata schema；
- [ ] environment snapshot interface；
- [ ] clean/fault evaluation pair；
- [ ] token boundary；
- [ ] paired metrics。

## R-M1：Local Mask

- [ ] fault prefix mask；
- [ ] clean behavior unchanged；
- [ ] gradient region test；
- [ ] B2 vs M2；
- [ ] credit diagnostic。

## R-M2：Paired Advantage

- [ ] valid pair filter；
- [ ] recovery group normalization；
- [ ] zero-variance handling；
- [ ] M2 vs M3；
- [ ] resource accounting。

## R-M3：Curriculum

- [ ] per-fault EMA；
- [ ] learning progress；
- [ ] minimum exploration；
- [ ] state checkpoint；
- [ ] equal-budget comparison。

## R-M4：Dynamic Reliability

- [ ] multiple overlapping tools；
- [ ] hidden reliability；
- [ ] drift；
- [ ] cost-aware verifier；
- [ ] routing metrics。

## R-M5：Paper Package

- [ ] main table；
- [ ] ablations；
- [ ] OOD；
- [ ] 3 seeds；
- [ ] figures；
- [ ] related work matrix；
- [ ] limitations；
- [ ] reproducibility checklist。

---

# 17. 研究风险与回退

## 17.1 Clean sibling 也有高方差

处理：

- 多个 clean suffix；
- paired difference；
- stratify by clean success；
- 分析方差，而非假设 clean reward 是真值。

## 17.2 Pair rollout 成本翻倍

处理：

- 固定总 trajectory 数；
- 共享 prefix token 或 KV cache（若框架支持）；
- 只对部分高价值样本分叉；
- 报告生成 token 而非只报告 step。

## 17.3 小模型不会恢复

处理：

- 故障文案结构化；
- 限制为临时错误；
- 允许 2–3 turn；
- 使用少量 recovery SFT warm-up 作为单独 baseline；
- 不用泄露正确答案的诊断。

## 17.4 与近期工作重叠

处理：

- 每周更新相关工作表；
- 保持窄 claim；
- 强调外生 fault intervention 与 prefix preservation；
- 把课程和风险目标视为辅助，而非主要 novelty。

## 17.5 核心方法失败

回退至工程项目：

- VerlTool 低算力复现；
- fault benchmark；
- Static-Fault；
- suffix-mask 负结果；
- 轨迹分析。

不得为了论文叙事隐瞒无效消融。

---

# 18. 找老师时的一页摘要模板

## 已完成

- 低预算 VerlTool Math-TIR 复现；
- clean/fault 工具可靠性评测；
- Static-Fault baseline；
- fault token 边界和 local mask。

## 观察

用真实结果填写：

- clean accuracy：`[ ]`；
- fault accuracy：`[ ]`；
- recovery rate：`[ ]`；
- Static-Fault 对 clean 的影响：`[ ]`；
- Local-Mask 初步效果：`[ ]`。

## 研究问题

> 当工具失败来自外部随机故障时，序列级 GRPO 是否错误惩罚故障前正确动作？共享 prefix 的 clean/fault sibling 是否能提供更可靠的局部信用？

## 希望老师帮助

- novelty 与相关工作判断；
- advantage estimator 设计；
- 强基线；
- 第二环境；
- 算力与投稿规划。

---

# 19. 参考资料

1. VerlTool repository: https://github.com/TIGER-AI-Lab/verl-tool  
2. VerlTool paper: https://arxiv.org/abs/2509.01055  
3. Fission-GRPO: https://arxiv.org/abs/2601.15625  
4. Credit Assignment Survey: https://arxiv.org/abs/2604.09459  
5. CRAFT: https://arxiv.org/abs/2606.29476  
6. veRL Agent Loop: https://verl.readthedocs.io/en/latest/advance/agent_loop.html  

