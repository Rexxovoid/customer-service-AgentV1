# 测评操作指南

更新：2026-10-30。本文负责测评操作，不重复定义业务规则；当前边界见[规则规范](../docs/rules.md)，职责和修改清单见[边界维护分工](../docs/boundary-maintenance.md)。原evals/README.md已更名为本文件。

## 案例与执行方式

复用现有create_order_agent与官方LangSmith evaluate，不操作日常runtime数据库、不覆盖旧实验。

| 案例文件 | 当前范围 | 用途 |
|---|---|---|
| `data/evaluation/cases.jsonl` | E01—E20，20条 | 主要业务、异常与真实RAG观察 |
| `data/evaluation/optimization_cases.jsonl` | E23—E30，8条 | 意图、补信息、日期和目标商品 |
| `data/evaluation/confirmation_cases.jsonl` | E31/E32，2条 | 句末标点确认、无效确认、幂等 |
| `data/evaluation/multiturn_cases.jsonl` | E33—E42，10条31轮 | 意图切换、权限、确认、更正、清零和持久幂等；全部待审批 |

审批状态以文件为准。主集E05/E06/E13最近修改后需重新审批；不因当前答案错误就把金标改成错误答案。测试题和参考答案不得导入知识库。

```powershell
uv sync --locked

# 格式校验，不调用模型。
uv run orderagent-eval validate

# 单条探索；不加--judge不调用评分模型。
uv run orderagent-eval run --ids E01
uv run orderagent-eval run --ids E01 --judge

# 重复与并发；每个案例的多轮仍顺序执行。
uv run orderagent-eval run --ids E05,E06,E13,E15 --repeat 3 --concurrency 2 --judge
uv run orderagent-eval run --cases data/evaluation/confirmation_cases.jsonl --repeat 3 --concurrency 2
uv run orderagent-eval run --cases data/evaluation/optimization_cases.jsonl --repeat 3 --concurrency 2

# 新多轮集：先校验，再单案例人工核对，最后批量重复。
uv run orderagent-eval validate --cases data/evaluation/multiturn_cases.jsonl
uv run orderagent-eval run --cases data/evaluation/multiturn_cases.jsonl --ids E33
uv run orderagent-eval run --cases data/evaluation/multiturn_cases.jsonl --repeat 3 --concurrency 2 --judge
```

repeat范围1—10，concurrency范围1—4，默认均1。LangSmith默认无云端上传；Langfuse由独立开关控制，见下文。模型调用仍产生供应商用量。裁判读取QIWEN_API_KEY、QIWEN_BASE_URL、QIANWEN；默认模型qwen3.8-flash，SDK重试1次、超时30秒、输出预算2000。裁判失败不改变业务硬规则结论。

## Langfuse追踪

设置 `LANGFUSE_ENABLED=true` 并配置三个 Langfuse 变量后，原测评命令即可附加官方回调追踪，无需 `--upload`。该参数仍专指现有 LangSmith 上传，未明确需要时不要添加。关闭 Langfuse 使用 `LANGFUSE_ENABLED=false`，默认即关闭；系统环境变量优先于 `.env`。

来源分别为 `eval_controlled_agent`（固定知识夹具）、`eval_live_agent`（实际Agent及配置知识源）、`eval_live_rag`（实际RAG检索/选择）。网页为 `web_chat`，终端为 `terminal_chat`。实际配置知识源及诊断一并记录，不因云端出现追踪就认定真实RAG通过。

每次案例执行有独立 `run_id` 和会话标识，保留 `case_id`。CLI为同案例分配线程安全的 `repetition_index`（从1开始的调度进入顺序，不是完成顺序）；直接调用 `execute` 未传入则为 `null`。每轮Agent调用有独立 `request_id` 及 `turn`；外层案例追踪也有自己的请求标识。`experiment.json` 记录Langfuse版本、启用状态及上传策略，原始配置凭据不保存。

只上传结构与白名单摘要，原始对话、完整状态、证据及裁判正文不上传；五维评分和人工审批仍以本地文件为准。RAG事件通过官方子observation补充，因为当前官方回调未实现项目自定义事件；这些摘要是收集后生成，摘要span本身耗时不是原始RAG阶段耗时，诊断的 `seconds` 才是可获取的原始阶段用时。

追踪服务不可用不改变硬规则和业务状态。上传异步进行，不保证网络异常或退出时记录完整；含无法脱敏事件、链接或状态正文的原始span不导出，异常分类另以安全摘要保留。因此网页中缺少某个span不能直接判为业务没有执行，应结合本地 `result.json`、`trace`、`events` 核对。

## 先审批案例，再评审结果

1. 打开对应JSONL，核对用户、订单、知识来源、初始状态、输入、expect、forbid、checks和reference。
2. 确认规格后将review_status改为approved；改过的案例重新pending。pending可以探索运行，不能称为人工验收通过。
3. checks是实际机器断言；expect/forbid中未转成断言的要求仍需人工或模型评审。
4. E18/E19需核对远端文档与前提，填写relevant_documents。只有approved且有相关文档金标才计算文档Hit@5；它不是答案正确率。真实RAG模式当前不运行语言裁判。

修改JSONL后校验格式并更新独立manifest字节数/SHA256。实验会另存实际数据快照和哈希，不拿旧清单冒充新版本。若VS Code提示磁盘较新，先Compare，不能用旧缓存覆盖新案例。

## 结果与人工评分

每次生成`evals/results/时间-随机编号/`，不覆盖历史：

| 文件 | 内容 |
|---|---|
| `experiment.json` | 代码/数据指纹、版本、模型配置和实验参数 |
| `results.jsonl` | 各次运行记录 |
| `summary.json` / `report.md` | 指标、失败归因、耗时和重复分布 |
| `E编号-运行编号/result.json` | 逐轮回答、检查、状态、工具、RAG与评分 |
| `human_review.csv` | 本次实验的独立人工评分表 |

先核对knowledge_source、initial_state及逐轮answer、checks.actual、state、operation、trace/events。建议先不看裁判分数，独立作人工判断。

填写human_review.csv：reviewer用别名；verdict为pass/fail/unjudgeable，空白表示未审；evidence填可核查回答、状态或原文；reason解释原因。run_id+turn+reviewer不可重复。预置turn=all为整场评分；可加逐轮行，但逐轮行不进入整场一致率。

五维按0—4评分：factual_correctness事实正确性、faithfulness知识忠实性、relevance问题相关性、actionability可操作性、expression表达质量。0明显错误、1重大缺陷、2部分满足、3基本满足、4充分满足；证据不足留空。语言均分不能抵消越权或错误放行。

```powershell
uv run orderagent-eval compare --result "evals/results/实际实验目录名"
```

compare生成comparison.json，不改人工表。输出人工与机器显式硬规则一致率、人工分减裁判分及证据；未审/无法判断不进入二元一致率。优先复核人工fail而机器pass、维度分差绝对值≥2的记录。重新运行是新实验，不能沿用旧run_id的审核。

## 隔离、指标与解释

- 每案例及重复运行独立SQLite/thread_id，同案例多轮共享。跨案例并发隔离不等于同订单并发安全；后者由共享临时台账的幂等回归单独验证。
- on_fail默认stop并记录未执行轮次；可配置continue或on_pass=stop。没有自由跳转或用户模拟器。
- 有效率=显式硬规则可判定pass/fail数÷计划数；失败仍是有效样本。另报有效样本通过率及计划通过比例。
- 无金标RAG为unjudgeable，目标异常为error，都保留计划分母；裁判覆盖率和各维有效数量单列。
- 总P50/P95是整案例目标耗时，含组装、检索及多轮，不含裁判，不是首次回复时间。逐轮耗时在turns[].seconds。
- 规则选择P50/P95来自selection_metrics，正常及异常可用用量均记录；用量缺失不等于0。
- 严格待确认消息可短路主LLM请求，标准AIMessage的name为deterministic_return_confirmation；工具节点、规则复核和官方限制钩子照常执行。实际LLM次数看llm轨迹，官方模型节点计数仍可能增加，不按节点数推算节省请求。没有待确认、载荷变更或非严格指令时不应出现该标记。
- 错误放行/误转人工必须有明确断言和人工金标，不能由通用通过率推算。少量样本不证明容量，受控规则不等于真实RAG端到端。

归因区分检索、规则选择/截断、工具路由/参数、状态/边界、目标异常及裁判错误。服务超时后正确转人工可以通过业务断言，同时仍保留依赖异常记录。

## 规则选择现场与配置对照

质量答复新增customer_reply，与原excerpt证据分离。人工核对步骤是否有依据、前提和警告是否保留，不因用户回复未显示分块ID或文档名扣分；原文和来源可在tool_results/artifact核对。新回复模式与旧的原文直出不同，比较实验时记录提示词/代码版本，不能把文案更短直接算作业务更正确。

裁判现使用官方流式累积，实验judge_parameters和judge_diagnostics的stream=true。诊断新增first_event_seconds、first_content_seconds、received_events、content_characters、max_event_gap_seconds；按服务端数据块统计，最后一块至超时的等待不计入相邻间隔。首块前超时相关指标为null；只有一个块时最大间隔null。无用量尾块时usage=null。

必须正常结束且stop、非空正文、五维结构校验通过才产生评分，部分JSON不算有效结果。不新增原始裁判正文日志。当前磁盘timeout30、max_retries1，和前次手动关闭重试的实验有差异，比较时应注明；流式持续返回可减少读取等待，但不能保证消除超时。[阿里云流式与JSON Mode说明](https://help.aliyun.com/en/model-studio/stream)。

裁判排查查看案例`result.json`中的`judge_diagnostics`：`error_type/error_detail`为最终异常，`cause_type/cause_detail`为直接底层异常；`total_seconds`包含SDK重试和解析。成功或响应后解析失败时可查看`finish_reason/usage/model/response_id/request_id`，无法获取的值为null。当前不记录每次重试，不能从总耗时推断准确尝试次数。

`judge_input`保存实际送评的精简输入（用户原文、最终回答、工具结果及其中知识依据、参考标准），不再重复发送消息历史、完整状态和硬规则检查结果。完整执行证据仍在`turns/messages/state`中供人工审查。评分输入版本为compact-v1，与旧评分输入不同，比较质量与耗时应注明版本。本次未改变裁判超时、重试和输出预算；历史报告不追改。

```powershell
$env:RAGFLOW_EVIDENCE_DEBUG = "true"
uv run orderagent --user-id DEMO-U-004 --trace
```

`evals/debug/rag_evidence/`中每次选择一份文件，保存实际输入、候选、预算、推理配置、可见raw_content、structured_result、parsing_error和匹配结果。终端摘要的debug_record指向文件；error/finish_reason/usage未暴露时为null。正文不作为新增custom事件输出，原有框架追踪仍按原配置管理。

默认关闭、最多20份后停止新增；已有文件可更新，需要手动清理。不保存隐藏推理，但可能含用户和知识正文，Git忽略不等于加密或权限隔离。排查完设为false并重启。诊断的完整限制见[rules.md](../docs/rules.md#rag诊断与本地记录)。

比较推理强度时，仅改变RAGFLOW_SELECTION_REASONING_EFFORT的high/low，预算保持4096，其余数据、提示词、状态不变。每组独立实验，记录正确率、截断、用量与耗时；低推理不保证稳定加速。阶段变更多时不能将所有差异归因于一个参数。

## 可选LangSmith上传与复现

```powershell
# 仅显式--upload才上传测评。
uv run orderagent-eval run --ids E01 --judge --upload
```

配置LANGSMITH_API_KEY后，官方Client创建新案例集，上传所选测试语句和预期；运行详情inputs/outputs/metadata隐藏，云端保留步骤、耗时、硬规则与维度反馈。本地保留完整脱敏记录。云端权限尚未实际验证，上传前核对数据。

不加--upload时，执行器显式关闭远程tracing，即使环境配置开启；普通CLI追踪不受此测评开关代替。结果记录提交/脏状态、代码/提示词/数据哈希、模型参数和可获取响应。远端知识无快照、动态模型别名变化时不能保证完全复现。本工具不是HTTP容量压测，也不提供真实平台退款成功率。
