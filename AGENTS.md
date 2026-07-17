# LiteVerlTool 项目协作规范

## 适用范围与规则优先级

- 本文件适用于整个仓库；若目标目录存在更近的 `AGENTS.md`，同时遵守其补充或收紧规则。
- 用户当次明确要求、当前设计文档和本文件发生冲突时，优先级依次为：用户要求 > `docs/design/` 当前设计 > 本文件 > 上游示例习惯。
- `docs/raw-design/` 只保存早期方案和研究备忘，不作为当前实现依据；当前主设计是 `docs/design/LITE_VERLTOOL_INTERNSHIP_PROJECT.md`。
- 不要求给上游已有的每个一级目录补建 `AGENTS.md` 或 `README.md`。只有项目自有模块出现独立约束时才增加局部规范，避免制造大量与上游冲突的文档改动。

## 项目定位

本项目是 VerlTool fork 上的工程增强，目标是快速形成可验证、可复现、可用于 Agentic RL 实习面试的项目：

```text
复现 Math-TIR 多轮工具训练链路
→ 建立可重放的工具故障 benchmark
→ 实现 Static-Fault Training
→ 分析 clean/fault 性能、恢复行为和资源成本
```

项目最低交付目标是设计文档定义的 L1：链路复现、可靠性评测、Static-Fault、指标与轨迹分析。post-fault local mask 是可选加分项，不得阻塞 L1 交付；CARE、环境快照、动态多工具可靠性和论文级实验矩阵不属于当前必做范围。

不得将以下内容包装为本项目成果：

- 上游 VerlTool/veRL 已有能力；
- 只复制但未验证的官方配置；
- 未实际运行的训练、评测或资源数据；
- Static-Fault 所没有解决的通用信用分配问题；
- 单次 seed 或随机波动产生的未经验证结论。

## 仓库边界

- `verl_tool/`：VerlTool 的 agent loop、Tool Server、trainer 适配和项目主要集成点。
- `verl/`：当前 fork 随仓库跟踪的上游 veRL 核心代码。优先使用现有扩展接口；只有无法在项目层实现时才修改，并记录原因、影响和回退方法。
- `examples/train/math_tir/`：上游 Math-TIR recipe 和参考脚本。可查阅和做必要安全修复，但项目配置与运行入口优先放入独立命名空间，避免持续散改上游示例。
- `benchmarks/*`：Git submodule。除非用户明确要求更新对应 benchmark，不修改其内容或指针。
- `docs/design/`：当前工程设计与验收标准。
- `docs/raw-design/`：只读历史方案，不作为完成状态或当前任务清单。
- 项目新增实现优先放在 `lite_verltool/`，配置放在 `configs/lite_verltool/`，运行入口放在 `scripts/lite_verltool/`，测试放在 `tests/lite_verltool/`。若实际接口要求放在上游目录，保持 patch 最小并在架构文档说明。

## 上游与版本管理

- 开始实质性实现前记录 fork 的 base commit、`verl/` 对应版本、benchmark submodule SHA、Python、PyTorch、CUDA、vLLM 和 Transformers 版本。
- `origin` 应指向个人 fork；建议另设 `upstream` 指向 `TIGER-AI-Lab/verl-tool`。不得在未检查差异的情况下直接合并或 rebase 浮动的上游 `main`。
- 上游同步必须单独进行，不与功能实现、实验配置或结果文档混在同一修改中。
- 不假定设计文档中的接口和配置键仍然有效；修改前以当前冻结 commit 的真实代码、Hydra 配置和脚本为准。
- 不单独更新 `verl/` 代码、CUDA 栈或 vLLM 版本来“顺手升级”。依赖升级需要明确动机、兼容性检查和 smoke test。

## 项目状态与事实源

项目状态统一维护在：

- `docs/status/PROJECT_COMPLETED.md`：有验证证据的已完成事实、产物、结果和最终决策。
- `docs/status/PROJECT_TODO.md`：当前任务、下一步、验收条件、阻塞和未解决风险。
- `docs/status/PROJECT_LOG.md`：需要长期追溯的环境问题、方向变化、实验复盘和关键取舍。

如果这些文件尚不存在，首次实施任务应先创建简洁模板。开始任务前读取适用的 `AGENTS.md`、当前设计、COMPLETED 和 TODO；仅在需要追溯时读取 LOG。不要只依赖对话历史或旧 raw-design。

状态记录规则：

- 代码写完但未验证不能记为完成；
- 完成项注明日期、路径、命令、环境和关键结果；
- GPU 实验注明 commit、配置、数据 subset、seed、GPU、wall-clock、peak VRAM、生成 token 和产物路径；
- 失败实验、负结果和阻塞必须如实记录，不得删除后只保留最好结果；
- TODO 完成后移入 COMPLETED，不创建平行的 `STATUS.md`、`TODO.md` 等事实源。

## 开发与环境约定

- 本地优先完成文档、静态检查、CPU 单测、mock Tool Server 和小型集成测试；远程 GPU 用于真实模型 rollout、backward、checkpoint 和训练评测。
- 依赖安装遵循冻结 commit 的上游安装指南，优先复用明确的容器或虚拟环境。不得为了形式统一强制迁移包管理器，也不得无目的升级 CUDA、PyTorch、vLLM 或 Transformers。
- 新配置不得依赖固定用户名、绝对 home 路径、固定端口或固定 `CUDA_VISIBLE_DEVICES`；通过参数或环境变量传入。
- 可公开的配置写入版本控制；密钥、服务器地址和个人运行参数只放 `.env` 或外部 secret manager，`.env.example` 只保留无敏感值的键名与说明。
- 训练产物、模型权重、数据缓存、WandB 目录和大体积轨迹不得直接提交。需要公开的小型结果应经过脱敏和体积检查。
- 修改前检查 `git status` 和相关 diff，保留用户已有改动；不得用重置、checkout 或格式化全仓库的方式覆盖无关内容。

## 安全要求

- 上游示例脚本在运行前必须审计硬编码凭据、用户路径、GPU 编号、端口和日志输出；发现疑似泄露的 key 时不得复制、打印或继续传播，应改为环境变量并提示轮换。
- `ipython_code` 会执行模型生成的任意代码。真实模型运行必须置于容器或可靠 sandbox 中，使用非 root 用户，并限制文件挂载、外网、CPU、内存、进程数和执行时间。
- 不得将字符串黑名单视为充分沙箱；不得让工具容器访问 SSH key、云平台凭据、WandB token 或宿主 home。
- 模拟 timeout 使用结构化错误 observation，不通过真实 sleep 消耗训练时间；真实服务故障测试与算法故障注入分开进行。
- 日志、错误 payload、轨迹和 checkpoint metadata 不得包含环境变量、token 或服务器凭据。

## Agent Loop 与 token 约束

- 保留真实的 action → Tool Server → observation → 后续生成闭环，不得用离线答案匹配冒充多轮工具执行。
- `response_mask=1` 只表示模型生成 token，observation 和 padding 必须为 0。
- 所有相关修改必须检查 `response_ids`、`response_mask`、attention mask 和可选 fault boundary 的长度与语义对齐。
- 仅有长度相等不代表 mask 正确；必须覆盖 action/observation 边界、retokenization、truncation、Unicode 和最后一轮终止。
- 若实现 local mask，fault boundary 必须在 token 构造时记录，不能从最终字符串二次猜测；全零 suffix mask 必须安全跳过。
- 修改 actor loss mask 时同步检查 policy loss、entropy、显式 KL、loss aggregation 和 GRPO group normalization，不能只验证一个张量相乘。

## 故障注入不变量

必须区分：

- `agent_error`：无工具标记、参数/语法错误、错误工具选择；
- `tool_execution_error`：Python 执行异常或代码自身超时；
- `injected_environment_fault`：对 parser 接受的工具请求人工施加的外生故障。

第一版主故障是结构化、可恢复的 `transient_error`。其语义必须满足：

```text
action_parse_valid = true
injected_fault = true
recoverable = true
done = false
```

此外必须满足：

- 使用稳定摘要而不是 Python 内置 `hash()` 决定故障，确保跨进程重放；
- 故障决策不依赖异步请求到达顺序；
- 每条轨迹最多注入一次；
- 保存 sample、rollout、tool turn、fault seed 和 token boundary；
- 不把 parser accepted 宣称为动作语义正确；
- agent/tool error 不进入 injected-fault 指标。

## 评测与实验纪律

- 最小主实验组为 Base、Clean GRPO、Static-Fault GRPO；不要在 baseline 稳定前同时展开多个故障率和算法变体。
- clean/fault 主评测使用相同题目集合，但独立采样只能称为 matched-question evaluation，不能称为严格 counterfactual rollout。
- Conditional Recovery 仅在 `clean_success ∧ fault_injected` 子集上计算；分母为零时输出 `N/A`。
- 计算 robustness gap 时 clean 和 fault 指标必须使用相同样本子集，不能混用全量 CleanAcc 与 triggered 子集 FaultAcc。
- 主比较固定 base checkpoint、数据 subset、rollout n、长度、max turns、训练 steps、verifier 和评测 seed，并同时报告生成 token 与 tool calls。
- GRPO 训练必须使用 `n>1`；记录奖励全同导致的 zero-variance group ratio。
- 先运行 1-step smoke，再运行 20–50 step profiling，之后才允许启动长训练或多 seed 实验。
- 正式实验不得只保存 WandB 图；必须保留可解析配置、聚合表和小型脱敏轨迹。

## 测试与验证

按风险选择最小但充分的验证：

- 文档或配置：检查格式、路径、配置解析和敏感信息；
- fault injector：概率边界、稳定重放、单轨迹一次、错误分类和 `done=false` 单测；
- Tool Server：clean、invalid、execution error、injected fault、状态隔离和资源清理集成测试；
- Agent Loop：真实或 mock 两轮交互、stop token、observation 回填、mask 和 stop reason；
- loss/mask：边界、截断、retokenization、全零 mask 和一步 backward；
- trainer：最小 rollout、optimizer step、checkpoint save/resume 和异常后服务清理。

不得用一次综合 GPU 训练代替分层测试。未运行某项验证时，在状态文件和最终回复中明确说明。

## Git 约定

- 仅在用户明确要求，或确实需要向远程 GPU 同步时执行 `git add`、`commit`、`push`；普通修改不自动提交。
- 提交前检查目标仓库、分支、`git status`、`git diff` 和 submodule 状态，不混入无关改动。
- 不提交 `.env`、API key、模型权重、数据缓存、运行日志、大型轨迹或未批准的 benchmark 指针变化。
- 未经明确授权，不改写历史、不强推、不删除用户分支、不执行破坏性清理。
- 提交信息使用中文，推荐 `模块：说明`，例如 `故障注入：增加可重放临时故障包装器`。
- 上游同步提交、项目功能提交、实验结果提交和纯格式修改尽量分开，便于审查和回退。

## 每次任务的固定汇报

完成代码、配置、文档、数据处理或实验后，最终回复必须明确包含：

- **目前进展**：实际修改、产物路径和已运行验证；
- **下一步**：最小可执行后续任务；
- **距离 L1 实习项目还差什么**：按链路复现、fault benchmark、Static-Fault、结果与展示说明缺口。

未运行的训练、测试或评测不得暗示已经完成；预期指标、验收阈值和真实实验结果必须明确区分。
