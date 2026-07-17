# Low-Cost VerlTool Reproduction and Reliability Enhancement

> 文档定位：工程复现、简历项目、实习投递保底线  
> 建议项目名：**LiteVerlTool：低算力多轮工具智能体强化学习与可靠性增强**  
> 最后核验日期：2026-07-16  
> 上游项目：TIGER-AI-Lab/verl-tool，当前主分支基于 veRL 0.6.0、vLLM 0.11.0 组织代码；上游已于 2026 年被 TMLR 接收。  
> 重要声明：本文中的指标目标均为验收阈值或待填结果，不代表已经获得实验结果。

---

## 0. 文档目的与使用方式

本文用于指导 Codex 在一个独立仓库或 VerlTool fork 中完成以下项目：

1. 在单卡或低 GPU 预算下跑通 VerlTool 的 Math-TIR 多轮 Agentic RL 链路；
2. 建立可复现的工具故障注入、轨迹记录和可靠性评测；
3. 实现两个低风险改进：静态故障训练与 post-fault suffix masking；
4. 形成可以用于简历、面试和后续科研升级的工程产物；
5. 即使论文型方法没有成功，项目也必须保持可运行、可测试、可展示。

本文是工程实施规格，不是论文方案。论文型升级见 `CARE_TOOL_RESEARCH_PLAN.md`。

### 0.1 Codex 执行协议

每次修改前必须：

1. 阅读仓库根目录 `AGENTS.md`；
2. 阅读 `docs/PROJECT_COMPLETED.md`；
3. 阅读 `docs/PROJECT_TODO.md`；
4. 阅读本文件中当前阶段、验收标准和禁止事项；
5. 检查上游 VerlTool commit，禁止在未记录 commit 的情况下依赖 `main` 的漂移行为。

每次修改后必须：

1. 运行与改动最相关的最小测试；
2. 将已完成事实、命令、产物和验证结果写入 `docs/PROJECT_COMPLETED.md`；
3. 将未完成事项、阻塞、风险和下一步写入 `docs/PROJECT_TODO.md`；
4. 不得把猜测写成已完成事实；
5. 不得修改训练算法与工具环境两个层面后只做一次综合测试，必须分层验证。

### 0.2 项目成功的三档定义

| 档位 | 达成条件 | 可用于 |
|---|---|---|
| E0：链路跑通 | 模型可生成工具动作、工具可执行、observation 回填、GRPO 可更新、checkpoint 可恢复 | 学习记录 |
| E1：工程项目 | 有低算力复现、可靠性评测、轨迹分析、Static-Fault 训练 | 简历与实习 |
| E2：增强项目 | 在 E1 上实现 suffix masking，完成严格消融并有稳定增益或有解释的负结果 | 强简历、找老师 |

---

# 1. STAR 法则：项目背景与目标

## S — Situation：背景

普通 RLVR 通常针对单轮生成：模型回答问题，验证器根据最终答案给出奖励。工具智能体则需要执行多轮闭环：

```text
用户任务
→ 模型推理并生成工具动作
→ 外部工具真实执行
→ 工具返回 observation
→ observation 写回上下文
→ 模型继续决策
→ 最终答案或终止
```

VerlTool 将这一过程统一到 veRL 训练框架中，提供：

- 多轮 action–observation Agent Loop；
- 统一 Tool Server API；
- 每条 trajectory 独立环境状态；
- Python、搜索、SQL、视觉、Web Search、SWE 等工具接入；
- observation token masking；
- 同步与异步 rollout；
- GRPO、DAPO 等训练 recipe。

官方 Math-TIR recipe 使用 Qwen2.5-Math-1.5B、DeepMath-103K 和 Python/IPython 工具。当前官方 `train_1.5b_grpo.sh` 默认仍是 4 GPU、每题 16 条 rollout、batch size 128、最长 response 3072、最多 10 turn 的高吞吐配置，不适合直接复制到单张消费级 GPU。

另一方面，真实工具并不可靠。timeout、临时异常、空返回和输出截断会使小模型提前终止、重复调用或陷入错误重试。仅报告 clean accuracy 无法反映 Agent 的部署可靠性。

## T — Task：任务

在有限算力下完成一个可复现、可评测的 VerlTool 工程项目：

1. 保留真正的多轮工具执行，而不是退化为单轮文本 GRPO；
2. 压缩 rollout、序列长度、训练数据和验证开销；
3. 记录 action、observation、turn、奖励和停止原因；
4. 构造确定性的外生工具故障；
5. 评测 clean/fault 性能差距和故障恢复率；
6. 实现简单、低风险的可靠性增强；
7. 输出完整 README、配置、脚本、结果表和轨迹案例。

## A — Action：行动

项目采用四阶段技术路线：

1. **Lite Reproduction**：缩小 Math-TIR recipe，跑通 Agent Loop、Tool Server、GRPO 和 observation mask；
2. **Reliability Evaluation**：加入可控故障 wrapper 与 clean/fault 配对评测；
3. **Static Fault Training**：训练时按固定概率注入故障，允许额外恢复 turn；
4. **Post-Fault Suffix Masking**：fault trajectory 只更新故障后的模型 token，避免外生故障污染故障前动作。

## R — Result：预期结果与验收

最低交付要求：

- 一条端到端训练命令可以启动 Tool Server 与 Trainer；
- 单卡配置能够完成 smoke run 和短训练；
- 至少产生 base、clean-GRPO、static-fault、suffix-mask 四组可比结果；
- 报告 clean accuracy、fault accuracy、conditional recovery rate、tool-call count；
- 可检索至少四类典型 trajectory；
- 所有实验记录模型、数据、seed、GPU、commit 和配置哈希；
- 即使改进无正增益，也必须给出失败分析和可复现证据。

完成后可用于简历的项目叙事：

> 基于 VerlTool/veRL 搭建低算力多轮 Tool Agent 强化学习系统，完成 Qwen2.5-Math-1.5B 的 Python 工具交互、GRPO 训练、observation masking 和异步轨迹记录；设计可控工具故障注入与恢复评测，并实现静态故障训练和故障后局部更新，系统分析 clean/fault 性能、恢复率和工具调用成本。

---

# 2. 原始 VerlTool 必须理解的技术要点

## 2.1 项目定位

VerlTool 是 Agentic RL 基础设施，不是一个独立的新优化算法。它的主要职责是：

```text
数据 → 多轮 rollout → 工具执行 → observation 回填
→ trajectory 奖励 → veRL 中的 GRPO/DAPO 更新
```

官方仓库强调以下设计：

- actor rollout 与环境交互解耦；
- Tool-as-Environment；
- 每个 trajectory 保存和恢复独立状态；
- 工具只需实现轻量 Python 定义；
- 原生多轮 Agent Loop；
- 可异步执行工具调用。

## 2.2 Agent Loop

当前主逻辑集中在：

```text
verl_tool/agent_loop/verltool_agent_loop.py
```

抽象流程：

```python
context = prompt
for turn in range(max_turns):
    action = llm.generate(context, stop=action_stop_tokens)
    if is_tool_action(action):
        observation = tool_server.execute(action, trajectory_id)
        context += action + observation
    else:
        break
return response_ids, response_mask, trajectory_metadata
```

Codex 必须确认当前 commit 的真实接口，不得只按伪代码修改。

## 2.3 Tool Server

核心职责：

- 解析 action；
- 选择工具；
- 加载 trajectory 环境；
- 执行 action；
- 返回 observation、done、valid 等信息；
- 更新或销毁 trajectory 状态。

官方训练脚本会启动类似服务：

```bash
python -m verl_tool.servers.serve \
  --host "$HOST" \
  --port "$PORT" \
  --tool_type ipython_code \
  --workers_per_tool 2
```

本项目禁止把工具服务简化为离线标准答案匹配。

## 2.4 Action stop token

Math-TIR 依靠特定 stop token 截断当前生成并触发工具。官方脚本将其配置为“三个反引号后接 `output`”。

模型生成这一标记后，rollout 暂停，前面的 Python 代码被送入工具服务。工具结果写回上下文后，模型继续生成。

## 2.5 Observation masking

完整 response 同时包含：

- 模型生成 token；
- 工具 observation token。

上游 `AgentLoopOutput` 使用：

```text
response_mask = 1：模型生成 token
response_mask = 0：工具返回 token
```

训练损失不能把 observation 当成模型动作。工程复现必须验证：

1. observation token 的 mask 为 0；
2. action token 的 mask 为 1；
3. retokenization 后长度对齐；
4. truncation 后 mask 不错位。

## 2.6 GRPO 信号

同一个 prompt 采样 `n > 1` 条 trajectory，得到组内奖励：

\[
A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}
\]

普通序列级 GRPO 会把同一 trajectory 的 advantage 广播到所有有效模型 token。工具 observation 由 `response_mask` 排除。

本工程项目不先重写 GRPO，只在 E2 阶段增加 fault 分支的局部 loss mask。

## 2.7 官方 Math-TIR recipe 的关键事实

官方当前脚本包含以下高成本设置：

```text
model = Qwen/Qwen2.5-Math-1.5B
n_gpus_per_node = 4
rollout.n = 16
train_batch_size = 128
max_prompt_length = 1024
max_response_length = 3072
max_obs_length = 512
max_action_length = 2048
max_turns = 10
mask_observations = true
rollout_mode = async
```

官方训练指南对低显存建议：

- `do_offload=True`；
- `enforce_eager=True`；
- `tensor_parallel_size=1`；
- `use_dynamic_bsz=False`；
- 减小 `ppo_micro_batch_size_per_gpu`；
- vLLM 卡住时降低工具 worker 和 `gpu_memory_utilization`。

### 安全要求

上游历史脚本可能包含不应复制的凭据或本地路径。Codex 必须：

- 删除任何硬编码 API key；
- 只从环境变量读取凭据；
- `.env` 不提交；
- 日志中不得打印 key；
- 训练脚本不得依赖特定用户目录或固定 GPU 编号。

---

# 3. 建议仓库结构

若基于 VerlTool fork，新增代码尽量放在独立命名空间，避免直接散改上游：

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   ├── lite_smoke.yaml
│   ├── lite_train.yaml
│   ├── fault_eval.yaml
│   ├── static_fault_train.yaml
│   └── suffix_mask_train.yaml
├── scripts/
│   ├── prepare_data.sh
│   ├── run_smoke.sh
│   ├── run_clean_train.sh
│   ├── run_fault_eval.sh
│   ├── run_static_fault_train.sh
│   ├── run_suffix_mask_train.sh
│   └── summarize_results.py
├── lite_verltool/
│   ├── faults/
│   │   ├── config.py
│   │   ├── injector.py
│   │   ├── registry.py
│   │   └── types.py
│   ├── masks/
│   │   ├── post_fault.py
│   │   └── validators.py
│   ├── metrics/
│   │   ├── recovery.py
│   │   ├── trajectory.py
│   │   └── aggregate.py
│   ├── analysis/
│   │   ├── classify_trajectories.py
│   │   └── render_trace.py
│   └── integration/
│       ├── agent_loop_hooks.py
│       └── reward_hooks.py
├── tests/
│   ├── unit/
│   │   ├── test_fault_injector.py
│   │   ├── test_mask_alignment.py
│   │   └── test_recovery_metrics.py
│   ├── integration/
│   │   ├── test_tool_server_clean.py
│   │   ├── test_tool_server_fault.py
│   │   └── test_agent_loop_two_turn.py
│   └── fixtures/
├── docs/
│   ├── PROJECT_COMPLETED.md
│   ├── PROJECT_TODO.md
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   └── FAILURE_TAXONOMY.md
└── results/
    ├── tables/
    ├── traces/
    └── figures/
```

如果上游接口不支持 hook，再做最小侵入式 patch；所有 patch 必须在 `docs/ARCHITECTURE.md` 记录原因、上游文件、修改点和回滚方法。

---

# 4. 低算力配置规划

以下是建议起点，不是官方保证配置。必须通过 smoke test 逐项调整。

## 4.1 Smoke 配置

```yaml
model: Qwen/Qwen2.5-Math-1.5B
train_samples: 32
val_samples: 16
n_gpus_per_node: 1
rollout_n: 2
train_batch_size: 2
ppo_mini_batch_size: 2
ppo_micro_batch_size_per_gpu: 1
max_prompt_length: 512
max_response_length: 512
max_action_length: 256
max_obs_length: 128
max_turns: 2
rollout_temperature: 1.0
gpu_memory_utilization: 0.20
do_offload: true
enforce_eager: true
use_dynamic_bsz: false
workers_per_tool: 1
val_n: 1
total_steps: 1
```

## 4.2 Pilot 配置

```yaml
train_samples: 512-2000
rollout_n: 4
train_batch_size: 4-16
max_response_length: 768-1024
max_action_length: 384-512
max_obs_length: 128-256
max_turns: 2-3
workers_per_tool: 1-2
val_n: 2
```

## 4.3 配置调优顺序

发生 OOM 时按以下顺序降低：

1. `gpu_memory_utilization`；
2. validation batch 和 `val_n`；
3. `max_response_length`；
4. `max_action_length`；
5. `rollout_n`；
6. `train_batch_size`；
7. 模型规模。

不要首先把 `max_turns` 降到 1，否则会破坏故障恢复场景。

---

# 5. 阶段化技术路线

## Phase 0：冻结上游与环境审计

### 任务

- 记录上游 repository URL、commit SHA、submodule SHA；
- 记录 Python、PyTorch、CUDA、veRL、vLLM、Transformers 版本；
- 检查官方脚本中的硬编码路径、GPU 和凭据；
- 生成 `docs/ARCHITECTURE.md`；
- 不做功能改动。

### 验收

- `python -m verl_tool.servers.serve --help` 正常；
- `python -m verl_tool.trainer.main_ppo --help` 或最小 import 正常；
- 版本表写入 `PROJECT_COMPLETED.md`；
- 所有 secret 已移除。

## Phase 1：Tool Server 单元与集成验证

### 任务

1. 启动 `ipython_code` 工具；
2. 提交合法 Python 动作；
3. 验证 observation；
4. 提交非法 Python 动作；
5. 验证错误 observation；
6. 测试 trajectory state 清理；
7. 测试两个并发 trajectory 不串状态。

### 验收

- clean action 正确执行；
- invalid action 被标记；
- timeout 有上限；
- 同 seed 行为可复现；
- 工具进程可优雅关闭。

## Phase 2：Agent Loop smoke run

### 任务

- 使用极小数据；
- 只执行 rollout，不更新或只更新一步；
- 保存完整 step record；
- 检查 action stop token；
- 检查 response mask。

### 必须输出

```json
{
  "trajectory_id": "...",
  "turns": 2,
  "actions": ["..."],
  "observations": ["..."],
  "response_mask_valid": true,
  "stop_reason": "...",
  "reward": 0.0
}
```

### 验收

- 至少有一条真实工具调用；
- observation 被回填后模型继续生成；
- mask 长度与 response token 数完全一致；
- 训练日志和轨迹日志可关联。

## Phase 3：低算力 clean baseline

### 实验组

- Base model：不训练；
- No-tool GRPO：可选，用于确认工具的实际价值；
- Clean VerlTool GRPO：主 baseline。

### 记录指标

- final answer accuracy；
- reward mean/std；
- tool use rate；
- average turns；
- average model tokens；
- average observation tokens；
- invalid action rate；
- wall-clock per training step；
- peak GPU memory。

### 验收

以下任一情况可判定“复现趋势成立”：

1. clean GRPO 相比 base 在固定小验证集上有稳定正增益；
2. 若受预算限制无显著增益，至少训练 reward、工具使用和轨迹质量呈可解释改善，且提供负结果分析。

禁止用单次随机波动声称复现成功。

## Phase 4：可靠性评测套件

### 4.1 故障类型

第一版只做外生故障：

```text
timeout
transient_error
empty_observation
truncated_observation
```

不要第一版加入错误数值或恶意 observation，它们属于工具信任与 misinformation 问题。

### 4.2 故障配置

```yaml
fault_enabled: true
fault_seed: 42
fault_probability: 0.10
fault_type_weights:
  timeout: 0.4
  transient_error: 0.3
  empty_observation: 0.2
  truncated_observation: 0.1
max_faults_per_trajectory: 1
inject_only_on_valid_action: true
```

### 4.3 条件评测

对同一个题目至少保存：

- clean rollout 是否成功；
- fault rollout 是否成功；
- 是否发生故障；
- 故障后是否继续行动；
- 是否恢复；
- 工具调用次数。

核心指标：

\[
\text{RecoveryRate}=P(\text{fault success}\mid\text{clean success})
\]

\[
\text{RobustnessGap}=Acc_{clean}-Acc_{fault}
\]

### 验收

- 故障可由 seed 重放；
- 故障只发生在合法 tool action 后；
- agent-caused error 与 injected fault 可区分；
- 聚合脚本能按故障类型输出结果。

## Phase 5：Static Fault Training

### 方法

训练时，以固定概率对合法工具调用注入一种外生故障，并允许后续恢复：

```text
正确工具动作
→ injected timeout
→ 模型继续生成
→ 重试、改写、切换或最终回答
```

奖励仍采用最终任务 verifier，第一版不设计手工“重试奖励”。

### 原因

直接奖励重试容易产生 reward hacking：模型可能主动制造多次调用。故障恢复应由最终成功约束。

### 对比

- Clean GRPO；
- Static-Fault 5%；
- Static-Fault 10%；
- 可选 Static-Fault 20%。

### 验收

- fault recovery 相比 clean-only baseline 有提升趋势；
- clean accuracy 不出现不可接受下降；
- 平均调用次数没有失控；
- 输出至少三类恢复轨迹与三类失败轨迹。

## Phase 6：Post-Fault Suffix Masking

### 动机

一条 fault trajectory：

```text
正确推理 → 正确调用 → 环境 timeout → 错误恢复 → 最终失败
```

普通序列级 advantage 可能同时惩罚正确调用和错误恢复。简单改进是：fault 分支只训练故障后模型 token。

### Mask 定义

已有：

```text
response_mask = 1：模型 token
response_mask = 0：observation token
```

新增：

```text
post_fault_mask = 0：故障发生前及故障 observation
post_fault_mask = 1：故障后的模型 token
```

最终 fault loss mask：

\[
M_t^{fault}=M_t^{response}\cdot M_t^{post-fault}
\]

### 实现要求

- clean trajectory 保持原有完整 `response_mask`；
- fault trajectory 使用 suffix mask；
- agent-caused syntax/parameter error 不使用外生故障 mask；
- `fault_token_boundary` 必须基于 token index 记录，不能依赖字符串二次猜测；
- truncation 后重新验证边界；
- 新增 mask 对齐单元测试。

### 关键消融

| 方法 | Fault 数据 | Prefix 训练 | Fault suffix 训练 |
|---|---:|---:|---:|
| Clean GRPO | 否 | 是 | 不适用 |
| Static Fault | 是 | 是 | 是 |
| Suffix Mask | 是 | 否（fault 分支） | 是 |

### 验收

- mask 单测覆盖 clean、单故障、多 turn、截断；
- suffix mask 不改变 clean trajectory loss；
- fault prefix 的梯度贡献为零；
- 与 Static Fault 在相同数据和 rollout 预算下比较。

---

# 6. 评测协议

## 6.1 主结果表

```text
| Method | Clean Acc | Fault Acc | Recovery Rate | Robustness Gap | Invalid Call | Avg Calls | Tokens |
```

## 6.2 故障分组表

```text
| Method | Timeout | Transient | Empty | Truncated | Worst Group |
```

## 6.3 效率表

```text
| Method | GPU Hours | Peak VRAM | Steps | Rollouts | Generated Tokens | Tool Calls |
```

## 6.4 轨迹行为分类

至少自动分类：

- successful_retry；
- successful_rewrite；
- immediate_abort；
- repeated_invalid_retry；
- redundant_retry；
- hallucinated_observation；
- no_tool_fallback；
- max_turn_termination。

## 6.5 公平性规则

所有主对比必须固定：

- base model；
- data subset；
- validation subset；
- rollout `n`；
- token budget；
- max turns；
- training steps；
- seed 列表；
- verifier。

如果某方法额外生成更多 rollout，必须报告总生成 token，不得只按 step 比较。

---

# 7. 测试计划

## 7.1 单元测试

- 故障 seed 可复现；
- 故障概率边界 0 和 1；
- fault type 权重归一化；
- 只对 valid action 注入；
- observation 截断长度；
- `post_fault_mask` 边界；
- recovery rate 分母为 0；
- clean/fault 结果聚合。

## 7.2 集成测试

- clean Tool Server；
- forced timeout Tool Server；
- 两 turn Agent Loop；
- clean/fault 同题评测；
- 一步 backward；
- checkpoint 保存与恢复；
- trainer 异常时工具服务被清理。

## 7.3 Smoke CI

CI 不跑完整 RL，只跑：

```text
imports
format/lint
unit tests
mock tool server
one-prompt agent loop with mock LLM
metrics aggregation
```

---

# 8. 风险、回退与停止条件

## 8.1 单卡无法训练

回退顺序：

1. 减小验证；
2. 减小长度；
3. 减小 rollout `n`；
4. 缩小训练集；
5. 尝试 LoRA，但必须单独验证权重同步到 rollout engine；
6. 使用云端双卡完成最终实验。

不得把“只做推理评测”描述成“完成 Agentic RL 训练”。

## 8.2 原版训练无增益

仍可保留：

- 完整链路复现；
- 显存和吞吐分析；
- 可靠性 benchmark；
- 轨迹诊断；
- 负结果与原因。

先确认 verifier、数据、mask 和工具调用率，再判断模型或预算不足。

## 8.3 Static Fault 没有效果

检查：

- 是否允许足够后续 turn；
- fault observation 是否对模型可理解；
- 是否大部分 clean 样本本来就失败；
- GRPO group 是否全部同奖导致零方差；
- 故障率是否过高；
- 小模型是否只会重复调用。

## 8.4 Suffix Mask 无明显提升

这仍是有效结论。必须比较：

- clean accuracy；
- fault recovery；
- prefix action probability；
- gradient norm；
- rollout 成本。

若无增益，不得继续堆叠课程学习来掩盖核心假设失败，应转入论文方案中的配对分支实验或保持工程项目定位。

---

# 9. 里程碑与 Codex 任务清单

## M0：仓库可运行

- [ ] 固定上游 commit；
- [ ] 环境安装；
- [ ] secret 清理；
- [ ] Tool Server smoke；
- [ ] 文档化依赖。

## M1：Agent Loop 跑通

- [ ] 真实 Python action；
- [ ] observation 回填；
- [ ] response mask 验证；
- [ ] step record 保存；
- [ ] 一步训练。

## M2：Clean baseline

- [ ] base eval；
- [ ] clean GRPO；
- [ ] 主指标；
- [ ] 轨迹案例；
- [ ] 资源记录。

## M3：Fault benchmark

- [ ] 四种外生故障；
- [ ] 可复现注入；
- [ ] clean/fault 条件评测；
- [ ] recovery rate；
- [ ] 行为分类。

## M4：简单改进

- [ ] Static Fault；
- [ ] suffix mask；
- [ ] 相同预算消融；
- [ ] 结果表；
- [ ] 失败分析。

## M5：简历交付

- [ ] README；
- [ ] 架构图；
- [ ] 一键命令；
- [ ] 主结果表；
- [ ] 轨迹可视化；
- [ ] 简历表述；
- [ ] 面试问答。

---

# 10. 简历与面试输出模板

## 10.1 简历项目标题

**LiteVerlTool：低算力多轮工具智能体强化学习与可靠性增强**

## 10.2 简历描述模板

仅在结果真实获得后填写数值：

- 基于 VerlTool/veRL 搭建 Qwen2.5-Math-1.5B 多轮 Python Tool Agent RL 链路，完成 vLLM rollout、Tool Server、GRPO、observation token masking、轨迹记录与低显存配置适配；在 `[GPU]` 上将官方高吞吐配置压缩至 `[配置]`，完成 `[步数/样本]` 训练。
- 构建 timeout、临时异常、空返回和截断 observation 的可控故障评测，提出 conditional recovery rate 与 robustness gap 分析，并自动分类重试、提前终止和重复调用等轨迹行为。
- 实现 Static-Fault 训练与 post-fault suffix masking，使 `[指标]` 从 `[A]` 提升至 `[B]`，同时 clean accuracy 变化 `[Δ]`；若未提升，改写为“系统验证其收益边界并定位失败原因”，不得伪造正结果。

## 10.3 面试必须能解释

- VerlTool 与普通 veRL 的区别；
- 为什么 observation token 不能参与策略梯度；
- GRPO 为什么需要组内多个 rollout；
- Agentic RL 的主要算力成本为什么是 rollout 而非单纯模型参数；
- 外生工具故障与模型生成错误的区别；
- 为什么不能直接奖励重试；
- suffix mask 的假设和局限；
- 如何保证故障评测可复现；
- 为什么该项目仍不是完整反事实信用分配方法。

---

# 11. 不做事项

第一阶段禁止：

- 接入在线 Web Search API；
- 复现 SWE-Bench；
- 同时引入 Python、SQL、Search 三种工具；
- 修改 GRPO、DAPO、奖励函数和 Agent Loop 四个层面；
- 使用 LLM-as-a-Judge 作为主要 verifier；
- 以单次 seed 的最优 checkpoint 报告结果；
- 将工程故障注入包装为论文级创新；
- 在没有 clean baseline 的情况下训练 fault 方法。

---

# 12. 参考资料

1. VerlTool official repository: https://github.com/TIGER-AI-Lab/verl-tool  
2. VerlTool paper: https://arxiv.org/abs/2509.01055  
3. Official training guide: https://github.com/TIGER-AI-Lab/verl-tool/blob/main/assets/docs/training_guide.md  
4. Official Math-TIR recipe: https://github.com/TIGER-AI-Lab/verl-tool/tree/main/examples/train/math_tir  
5. Official 1.5B GRPO script: https://github.com/TIGER-AI-Lab/verl-tool/blob/main/examples/train/math_tir/train_1.5b_grpo.sh  
6. veRL Agent Loop documentation: https://verl.readthedocs.io/en/latest/advance/agent_loop.html  

