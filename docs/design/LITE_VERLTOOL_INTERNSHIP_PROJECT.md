# LiteVerlTool：面向快速复现、简单改进与实习求职的项目设计

> 文档定位：可执行的工程设计与实验规格  
> 目标优先级：快速复现 > 可靠性评测 > 简单改进 > 论文探索  
> 上游项目：TIGER-AI-Lab/verl-tool  
> 推荐基础模型：Qwen2.5-Math-1.5B  
> 推荐训练资源：4 × 24GB GPU，64–128GB RAM  
> 状态声明：本文中的指标均为验收定义或待填结果，不代表已经完成实验。

---

## 1. 项目结论

本项目应定位为一个 **Agentic RL 工程复现与工具可靠性增强项目**，而不是从第一天开始追求论文创新。

最合理的主线是：

```text
复现 VerlTool Math-TIR 多轮训练链路
→ 建立可重放的工具故障 benchmark
→ 实现 Static-Fault Training
→ 分析 clean/fault 性能与恢复行为
→ 形成代码、测试、结果表和轨迹可视化
```

该范围能够同时体现：

- 对 veRL、GRPO、vLLM rollout 和 FSDP 的理解；
- 对多轮 Agent action–observation 数据流的理解；
- 对工具服务、异步执行、轨迹状态和故障语义的工程能力；
- 对实验公平性、资源核算和失败分析的研究素养。

### 1.1 成功等级

| 等级 | 达成条件 | 求职价值 |
|---|---|---|
| L0：链路复现 | 工具调用、observation 回填、mask、一步 GRPO、checkpoint 恢复全部通过 | 证明能够运行和理解框架 |
| L1：完整工程 | L0 + fault benchmark + Static-Fault + 指标与轨迹分析 | 实习项目主交付 |
| L2：加分实验 | L1 + post-fault local mask 严格消融，允许正结果或可解释负结果 | 强简历、联系导师 |

本项目的最低成功标准是 **L1**。L2 不应阻塞 README、简历和演示交付。

---

## 2. 范围与非目标

### 2.1 必做范围

1. 固定一个 VerlTool commit 及其 veRL submodule commit；
2. 复现 Math-TIR 的 Python 工具多轮 rollout；
3. 验证 observation token 不参与策略损失；
4. 完成 Qwen2.5-Math-1.5B 的小规模 clean GRPO；
5. 建立确定性、可重放的外生故障注入；
6. 区分 agent error、tool execution error 和 injected fault；
7. 完成 clean/fault 评测和 Static-Fault Training；
8. 输出资源、指标、轨迹案例、测试和复现命令。

### 2.2 可选范围

- post-fault local loss mask；
- 第二种故障类型；
- LoRA 与 full-parameter 的资源对比；
- 一个简单的交互式轨迹查看器；
- 核心结果的 3-seed 验证。

### 2.3 明确不做

- CARE 的严格 clean/fault 环境分叉；
- IPython kernel 的通用 snapshot/clone；
- curriculum、CVaR、动态多工具可靠性；
- Web Search、SWE-Bench、视觉工具；
- 多种 RL 算法同时比较；
- LLM-as-a-Judge 主奖励；
- 为追求正结果而不断增加方法组件。

严格 counterfactual branching 需要可复制的环境状态和重新定义的 GRPO group normalization，工程风险显著高于本项目目标，应在 L1 完成后另立研究计划。

---

## 3. 技术正确性与边界

### 3.1 VerlTool 数据流

项目必须保留真实多轮闭环：

```text
prompt
→ LLM 生成 action token
→ action stop token 触发 Tool Server
→ Tool Server 解析并执行 Python
→ observation token 写回上下文
→ LLM 基于 observation 继续生成
→ verifier 计算终局奖励
→ GRPO 更新 actor
```

禁止把工具执行替换成离线标准答案匹配，否则不能称为 VerlTool 多轮复现。

### 3.2 Observation masking

完整 response 同时包含模型 token 和环境 token：

```text
response_mask = 1：模型生成 token
response_mask = 0：工具 observation 和 padding token
```

必须验证：

- action token 的 mask 为 1；
- observation token 的 mask 为 0；
- `len(response_ids) == len(response_mask)`；
- action/observation 连接处没有 off-by-one；
- retokenization 和 truncation 后 token 来源没有错位；
- policy loss、entropy 和显式 KL 使用一致的有效 token 区域。

仅验证长度相等不够。retokenization 改变 token 数量时，不能简单复制最后一个 mask 值作为正确性证明。

### 3.3 GRPO 的最低要求

同一 prompt 需要多个 rollout，才能形成组内相对优势：

\[
A_i=\frac{R_i-\mu_g}{\sigma_g+\epsilon}
\]

工程上需要注意：

- 训练时 `rollout_n` 必须大于 1；
- `n=2` 只适合 smoke，pilot 建议从 `n=4` 开始；
- 如果同组奖励完全相同，GRPO advantage 可能全部为零；
- 小模型和稀疏终局奖励组合容易产生大量零方差 group；
- 必须记录 zero-variance group ratio，而不能只看 mean reward。

### 3.4 本项目不声称什么

Static-Fault 是可靠性数据增强，不是新的信用分配算法。post-fault local mask 即使有效，也只能声称：

> 去除了 fault prefix token 对 actor objective 的直接损失项，并测试其对故障恢复的影响。

它不能保证后缀损失不会通过共享 Transformer 参数改变 prefix 策略，也不能自动消除 fault reward 对 GRPO 组内均值和方差的影响。

---

## 4. 故障模型与正确语义

### 4.1 三类失败必须分开

| 类别 | 示例 | 是否进入 injected-fault 指标 |
|---|---|---:|
| Agent error | 无工具标记、参数错误、语法错误、选择错误工具 | 否 |
| Tool execution error | Python 本身抛异常、代码运行超时 | 否 |
| Injected environment fault | 对合法工具请求人工返回临时不可用或截断 observation | 是 |

“通过 action parser”不等于“代码可执行”，更不等于“语义正确”。工程版只承诺对 parser 接受的调用注入；涉及“正确工具动作”的分析还必须额外以 clean success 或任务检查作为条件。

### 4.2 Recoverable fault 返回语义

人工注入的临时故障必须允许 Agent 继续行动：

```json
{
  "action_parse_valid": true,
  "tool_execution_attempted": false,
  "tool_execution_success": false,
  "injected_fault": true,
  "fault_type": "transient_error",
  "recoverable": true,
  "done": false,
  "observation": "[TOOL_ERROR transient] Service temporarily unavailable. Retry is allowed."
}
```

不能沿用真实服务失联时常见的 `done=true, valid=false` 语义，否则轨迹直接终止，模型没有恢复机会。

第一版 timeout 应当是**模拟 timeout observation**，而不是真的等待若干秒。真实 sleep 只会浪费 rollout 时间，并把服务性能问题混入算法实验。

### 4.3 第一版故障类型

主实验只使用一种结构化、可恢复的故障：

```text
transient_error
```

在主链路稳定后，再依次加入：

1. empty observation；
2. truncated observation；
3. simulated timeout。

错误数值、恶意 observation 和永久服务失效不属于本项目范围。

### 4.4 确定性注入

禁止依赖异步请求到达顺序决定是否注入。建议使用稳定键：

```text
fault_key = sha256(sample_id, rollout_id, tool_turn, fault_seed)
```

注入器必须支持：

- `fault_probability=0` 和 `1`；
- 固定 seed 重放相同故障决策；
- 每条轨迹最多一次故障；
- 只在 action parse valid 后触发；
- 保存 fault turn、token boundary 和原始请求 metadata。

可严格重放的是**故障决策**。异步 vLLM rollout 和整个训练过程不承诺 bit-exact。

---

## 5. 评测设计

### 5.1 两种评测模式

#### Clean evaluation

关闭故障注入，测量模型正常任务能力。

#### Matched-question fault evaluation

使用与 clean 相同的题目集合，在满足注入条件的首次合法工具调用后强制注入一次故障。

这是“同题配对评测”，不是严格共享模型 prefix 的 counterfactual rollout。两次 rollout 的采样差异必须被视为噪声。

### 5.2 主指标

设：

- `C=1`：clean rollout 最终成功；
- `I=1`：fault rollout 实际成功注入故障；
- `F=1`：注入后 fault rollout 最终成功。

主指标定义为：

\[
CleanAcc=P(C=1)
\]

\[
TriggeredFaultAcc=P(F=1\mid I=1)
\]

为了在相同样本子集上计算性能差距，定义：

\[
MatchedCleanAcc=P(C=1\mid I=1)
\]

\[
ConditionalRecovery=\frac{\sum 1[C=1\land I=1\land F=1]}{\sum 1[C=1\land I=1]}
\]

\[
MatchedRobustnessGap=MatchedCleanAcc-TriggeredFaultAcc
\]

不能直接用全量 `CleanAcc` 减去只在触发子集上计算的 `TriggeredFaultAcc`，因为二者分母不同。报告中同时保留全量 CleanAcc 和 matched gap。

如果 `C=1 ∧ I=1` 的样本数为零，Conditional Recovery 必须输出 `N/A`，不能输出 0。

### 5.3 行为与成本指标

- tool-use rate；
- injection coverage；
- recovery-attempt rate；
- immediate-abort rate；
- successful retry rate；
- repeated retry rate；
- invalid action rate；
- average tool calls；
- average model/observation tokens；
- zero-variance group ratio；
- wall-clock、peak VRAM 和生成 token 数。

### 5.4 轨迹分类

第一版自动分类：

```text
clean_success
successful_retry
success_without_retry
immediate_abort_after_fault
repeated_retry
invalid_retry
max_turn_termination
no_fault_triggered
```

每类至少保存 2–3 条匿名化轨迹用于 README 或面试展示。

---

## 6. 简单改进：Static-Fault Training

### 6.1 方法

训练时，以固定概率在首次合法工具调用后注入一次 recoverable transient fault：

```text
合法 Python action
→ transient fault observation
→ 模型继续生成
→ 重试、改写代码或使用已有推理完成任务
→ 终局 verifier 奖励
```

不增加“执行重试即可得分”的 shaping reward。否则模型可能通过无意义重复调用进行 reward hacking。

### 6.2 最小实验组

| ID | 方法 | 是否训练 | 训练故障率 | 用途 |
|---|---|---:|---:|---|
| B0 | Base model | 否 | 0 | 初始能力 |
| B1 | Clean GRPO | 是 | 0 | 复现 baseline |
| M1 | Static-Fault GRPO | 是 | 10% | 简单改进 |

开发阶段不同时运行 5%、10%、20% 三组。只有 M1 明显过弱或过强时，才补一个故障率消融。

所有训练方法必须固定：

- base checkpoint；
- train/validation subset；
- rollout `n`；
- optimizer 和学习率；
- max turns 和长度；
- training steps；
- verifier；
- 主评测题目和 fault seed。

同时报告总生成 token。仅固定 step 不能保证实际计算预算相同。

### 6.3 成功判定

Static-Fault 成功不要求得到夸张提升。满足以下证据链即可形成可信工程结果：

1. fault benchmark 能稳定降低 clean baseline 的表现；
2. Static-Fault 提高 recovery 或降低 immediate abort；
3. clean accuracy 没有明显不可接受的退化；
4. 平均工具调用次数没有失控；
5. 结果能由轨迹案例解释。

若没有增益，先检查 injection coverage、剩余 turn、zero-variance group、故障文案和 base clean success，再将结果如实写成负结果。

---

## 7. 可选加分：Post-Fault Local Mask

该实验只有在 B1 和 M1 都稳定后才开始。

对 fault trajectory 定义：

```text
local_mask = 0：故障 observation 结束位置之前的 token
local_mask = 1：故障之后由模型生成的 token
```

最终 fault loss token 区域：

\[
M_t=M_t^{response}\cdot M_t^{local}
\]

实现要求：

- clean trajectory 完全保持原 response mask；
- fault boundary 在构造 token 时记录，不从最终字符串猜测；
- action parser error 和 Python exception 不启用 local mask；
- 全零 suffix mask 的轨迹被安全跳过；
- policy、entropy、显式 KL 的 mask 行为有测试；
- 明确当前 GRPO group normalization 是否仍使用原始 fault reward；
- 比较 M1 和 local-mask 方法时固定 rollout/token 预算。

允许的结论是“局部化 fault trajectory 的直接 policy-loss token 区域”，不能声称已经解决 Agentic RL 的反事实信用分配。

---

## 8. 阶段化实施计划

### Phase 0：冻结与环境审计

任务：

- 选择并记录上游 commit 和 submodule commit；
- 记录 Python、CUDA、PyTorch、veRL、vLLM、Transformers 版本；
- 清除硬编码 token、用户目录、GPU 编号和 WandB 配置；
- 创建项目状态文档和实验清单；
- 在隔离容器中运行模型生成代码。

验收：

- Tool Server 和 Trainer 可以 import；
- 上游 commit 不再漂移；
- 日志与配置不含凭据；
- 生成代码无法访问宿主 SSH key 和云凭据。

### Phase 1：Tool Server

任务：

- 合法 Python action；
- 语法错误和运行异常；
- 两条 trajectory 状态隔离；
- timeout 和资源限制；
- trajectory 清理；
- transient fault wrapper。

验收：

- clean observation 正确；
- agent/tool/fault 三类错误可区分；
- injected fault 返回 `done=false`；
- 同一 fault key 可重放；
- 服务结束后无残留 kernel。

### Phase 2：Agent Loop 与 mask

任务：

- 运行一条真实两轮工具轨迹；
- 保存 action、observation、token ids、mask 和 stop reason；
- 验证 retokenization/truncation；
- 运行一步 backward。

验收：

- 至少一次 observation 回填后模型继续生成；
- response ids 与 mask 对齐；
- observation token 无直接策略损失；
- loss 有限且 optimizer step 成功。

### Phase 3：Clean baseline

任务：

- base clean/fault evaluation；
- clean GRPO pilot；
- checkpoint 保存和恢复；
- 记录显存、吞吐和 zero-variance group。

验收：

- clean GRPO 可重复启动；
- checkpoint 恢复后可继续训练；
- 至少得到可解释的 reward/trajectory 变化；
- 不用单次验证波动声称模型显著提升。

### Phase 4：Fault benchmark

任务：

- matched-question clean/fault evaluation；
- 聚合主指标；
- 自动轨迹分类；
- 生成结果表和案例。

验收：

- injection coverage 可报告；
- Conditional Recovery 分母正确；
- fault seed 可重放；
- agent error 不被算作 injected fault。

### Phase 5：Static-Fault

任务：

- 训练 M1；
- 与 B1 做相同预算比较；
- clean/fault 双评测；
- 分析工具调用成本和失败模式。

验收：

- 结果表包含 B0/B1/M1；
- 至少一个 seed 完整跑通；
- 有正结果或完整负结果分析；
- README、复现命令和轨迹展示可独立阅读。

### Phase 6：可选 Local Mask

仅在 Phase 5 完成后执行。该阶段不阻塞求职交付。

---

## 9. 算力与配置规划

### 9.1 推荐云端资源

| 用途 | GPU | CPU/RAM | 说明 |
|---|---|---|---|
| 工具和单元测试 | 无 GPU 或本机 8GB | 8 核/16GB | 不启动正式 RL |
| Agent Loop smoke | 1–2 × 24GB | 16 核/32–64GB | 极小 batch，一步更新 |
| 1.5B pilot | 4 × 24GB | 24–32 核/64–128GB | 推荐性价比配置 |
| 减少 OOM 调参 | 4 × 40/48GB | 32 核/128GB | 更适合 1024–2048 token |

推荐优先租 4 × RTX 4090 24GB 或 A5000 24GB 级别实例；希望减少 OOM 调参时可选 A6000/L40S 48GB。除非要扩大序列、rollout 或实验矩阵，否则 4 × 80GB 对 1.5B 项目不是必要条件。

### 9.2 Smoke 规划值

以下是启动规划，不是保证可直接映射到任意 commit 的最终 Hydra 配置：

```yaml
model: Qwen/Qwen2.5-Math-1.5B
gpus: 2-4
train_prompts: 8-32
val_prompts: 8-16
rollout_n: 2
train_batch_size: 2-4
ppo_micro_batch_size_per_gpu: 1
max_prompt_length: 512
max_response_length: 512
max_action_length: 256
max_obs_length: 128
max_turns: 2
total_steps: 1
workers_per_tool: 1
offload: true
enforce_eager: true
dynamic_batching: false
```

### 9.3 Pilot 规划值

```yaml
gpus: 4
train_prompts: 512-2000
rollout_n: 4
train_batch_size: 8-16
ppo_micro_batch_size_per_gpu: 1
max_prompt_length: 512
max_response_length: 768-1024
max_action_length: 384-512
max_obs_length: 128-256
max_turns: 2-3
workers_per_tool: 1-2
validation_n: 1-2
```

pilot 配置必须再明确 `total_steps` 或 epochs；否则不能估算总成本。

### 9.4 OOM 调整顺序

1. 降低 validation 数量和频率；
2. 降低 response/action 长度；
3. 降低 vLLM 并发序列数和 token batching；
4. 降低 rollout `n`，但保持 `n > 1`；
5. 降低 train batch；
6. 开启参数/优化器 offload；
7. 尝试 LoRA，并验证 adapter 到 rollout engine 的同步；
8. 最后才更换更小模型。

`gpu_memory_utilization` 不能盲目降到极低；过低可能导致 rollout engine 没有足够空间加载权重或 KV cache。每次调整只改变一个主变量并记录结果。

### 9.5 资源核算

每个实验至少报告：

```text
GPU 型号与数量
峰值显存
CPU RAM 峰值
训练 wall-clock
GPU-hours = GPU 数量 × wall-clock hours
训练 steps
有效 prompts
总 trajectories
总生成 token
总 tool calls
平均 tokens/trajectory
```

正式租用前先运行 20–50 step profiling，再按真实吞吐估算 B1/M1 总成本。

---

## 10. 安全与可复现性

Python 工具执行的是模型生成的任意代码。必须：

- 使用容器或可靠 sandbox；
- 使用非 root 用户；
- 不挂载宿主 home、SSH key 和云凭据；
- 默认关闭工具容器外网；
- 限制执行时间、内存、CPU 和进程数；
- 不依赖字符串黑名单作为唯一防护；
- 工具日志不打印环境变量；
- `.env`、checkpoint credential 和 WandB key 不提交。

复现实验必须记录：

- repository 和 submodule SHA；
- 完整解析后的配置；
- 数据文件 hash 和 subset ID；
- model revision；
- fault seed 和训练 seed；
- 容器镜像或 lockfile；
- GPU、driver 和 CUDA 版本。

---

## 11. 建议仓库结构

```text
.
├── README.md
├── AGENTS.md
├── configs/
│   ├── smoke.yaml
│   ├── clean_train.yaml
│   ├── fault_eval.yaml
│   ├── static_fault_train.yaml
│   └── local_mask_train.yaml          # optional
├── scripts/
│   ├── prepare_data.sh
│   ├── run_smoke.sh
│   ├── run_clean_train.sh
│   ├── run_fault_eval.sh
│   ├── run_static_fault_train.sh
│   └── summarize_results.py
├── lite_verltool/
│   ├── faults/
│   │   ├── config.py
│   │   ├── injector.py
│   │   └── types.py
│   ├── metrics/
│   │   ├── recovery.py
│   │   └── aggregate.py
│   ├── traces/
│   │   ├── classify.py
│   │   └── render.py
│   └── masks/
│       └── post_fault.py              # optional
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── design/
│   ├── PROJECT_COMPLETED.md
│   ├── PROJECT_TODO.md
│   ├── ARCHITECTURE.md
│   └── EXPERIMENTS.md
└── results/
    ├── tables/
    ├── traces/
    └── figures/
```

优先通过独立 wrapper、hook 和 metadata 扩展上游。只有现有接口无法支持时，才最小化修改 VerlTool/veRL 文件，并在 `ARCHITECTURE.md` 记录 patch 原因和位置。

---

## 12. 测试计划

### 单元测试

- fault probability 0/1；
- 相同 fault key 重放；
- 只对 parse-valid action 注入；
- injected fault 的 `done=false`；
- 每轨迹最多一次故障；
- recovery 分母为零；
- clean/fault 聚合；
- token boundary 和 local mask（可选）。

### 集成测试

- clean Tool Server；
- transient fault Tool Server；
- 两条并发 trajectory 不串状态；
- 两轮 Agent Loop；
- observation mask；
- 一步 backward；
- checkpoint save/resume；
- Trainer 异常后清理工具服务。

### CI 范围

CI 不执行正式 GPU RL，只运行：

```text
imports
lint/format
unit tests
mock Tool Server
mock LLM 两轮 Agent Loop
metric aggregation
trace classification
```

---

## 13. 两周执行节奏

| 时间 | 目标 |
|---|---|
| Day 1–2 | 冻结上游、环境安装、凭据和 sandbox 审计 |
| Day 3–4 | Tool Server、两轮 Agent Loop、mask 测试 |
| Day 5 | 一步 GRPO、checkpoint、资源 profiling |
| Day 6–7 | Base eval 与 clean GRPO pilot |
| Day 8–9 | Fault injector、matched-question eval、指标 |
| Day 10–11 | Static-Fault pilot 与结果比较 |
| Day 12 | 轨迹分类、失败分析、补最小实验 |
| Day 13–14 | README、结果表、架构图、简历和面试材料 |

如果 Day 7 前仍未完成一步稳定 GRPO，应停止算法改进，集中修复环境、mask、reward 和 checkpoint。

---

## 14. 最终交付物

### 代码

- 一键 smoke 命令；
- clean training 命令；
- fault evaluation 命令；
- Static-Fault training 命令；
- 单元与集成测试；
- 配置和结果聚合脚本。

### 结果

主表：

```text
| Method | Clean Acc | Matched Clean Acc | Triggered Fault Acc |
| Conditional Recovery | Matched Robustness Gap | Injection Coverage |
| Avg Calls | Tokens | GPU Hours |
```

诊断表：

```text
| Method | Immediate Abort | Successful Retry | Repeated Retry |
| Invalid Call | Zero-variance Groups | Peak VRAM |
```

### 展示

- 一张架构图；
- 一张 clean/fault 对比图；
- 3–6 条典型轨迹；
- 一份资源与失败分析；
- 可复制的环境和命令。

---

## 15. 简历与面试表述

### 项目标题

**LiteVerlTool：多轮工具智能体强化学习与故障恢复评测**

### 简历模板

只有获得真实结果后才能填写数值：

- 基于 VerlTool/veRL 复现 Qwen2.5-Math-1.5B 多轮 Python Tool Agent GRPO，完成 vLLM rollout、FSDP 训练、observation token masking、轨迹记录与 checkpoint 恢复，并在 `[GPU 配置]` 上完成 `[steps/tokens]` 的低预算训练。
- 构建可重放的 transient tool fault benchmark，区分 Agent 错误、工具执行错误和外生注入故障，评测 clean accuracy、triggered fault accuracy、conditional recovery 与工具调用成本。
- 实现 Static-Fault Training，使 `[真实指标]` 从 `[A]` 变化至 `[B]`，并通过重试、提前终止、重复调用等轨迹分类分析方法收益与失败边界。

若没有正增益，应改写为：

> 完成相同 token 预算下的可靠性消融，定位故障覆盖率、零方差 GRPO group 和恢复 turn 不足等失败原因。

### 面试必须能解释

- VerlTool 与普通单轮 RLVR 的区别；
- vLLM rollout 与 FSDP actor 如何协作；
- 为什么 observation token 不能直接参与策略损失；
- GRPO 为什么需要同 prompt 多 rollout；
- 为什么全同奖励 group 没有有效相对信号；
- agent error、tool error 和 injected fault 的区别；
- 为什么 injected transient fault 必须 `done=false`；
- 为什么不能直接奖励重试；
- matched-question eval 为什么不是严格 counterfactual；
- Static-Fault 和 local mask 各自解决什么、没有解决什么。

---

## 16. Go/No-Go 规则

### Gate A：复现成立

继续 fault 项目的条件：

- 工具真实执行；
- observation 回填；
- mask 正确；
- 一步 backward；
- checkpoint 恢复。

### Gate B：benchmark 有效

继续 Static-Fault 的条件：

- injection coverage 足够；
- fault 确实对 baseline 造成可测影响；
- Agent 至少有一个后续 turn；
- 指标分母和错误分类正确。

### Gate C：是否做 local mask

只有在以下条件满足时进入可选实验：

- B1/M1 已完成；
- fault token boundary 可靠；
- Static-Fault 暴露出合理的恢复问题；
- 仍有算力和时间预算。

否则直接完成求职交付，不让研究扩展拖垮工程项目。

---

## 17. 参考入口

1. VerlTool repository: <https://github.com/TIGER-AI-Lab/verl-tool>
2. VerlTool Math-TIR recipe: <https://github.com/TIGER-AI-Lab/verl-tool/tree/main/examples/train/math_tir>
3. VerlTool training guide: <https://github.com/TIGER-AI-Lab/verl-tool/blob/main/assets/docs/training_guide.md>
4. veRL Agent Loop: <https://verl.readthedocs.io/en/latest/advance/agent_loop.html>
5. veRL LoRA RL guide: <https://verl.readthedocs.io/en/latest/advance/ppo_lora.html>

实现前必须把这些入口解析到固定 commit，而不是长期依赖浮动的 `main`。
