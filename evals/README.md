# 独立自动测评与人工核对

2026-10-27：当前E06验证质量多轮、第4—5次建议/第6次人工、新问题和结束清零；E13验证独立退货计数。旧实验属于旧规则，当前规格必须重新人工审核。结果增加knowledge_source和initial_state，人工对比时先核对知识来源与初始数据库；普通CLI使用持久数据库，自动案例使用独立新库。

复用现有create_order_agent与官方LangSmith evaluate。不改日常启动方式，不操作runtime业务数据库。22条案例位于data/evaluation/cases.jsonl（原20条＋左键E21/空格键E22）。当前为命令行工具，没有网页审批工作台。

## 安装与运行

```powershell
uv sync
uv run orderagent-eval validate  检测用例校验
uv run orderagent-eval run --ids E01 --judge 检测单条调试
uv run orderagent-eval run --judge 全量运行
uv run orderagent-eval run --ids E06,E10,E11,E13,E15,E20 --judge 小批量回归
uv run orderagent-eval run --repeat 3 --concurrency 2 --judge 正式评测
```

裁判读取.env现有QIWEN_API_KEY、QIWEN_BASE_URL、QIANWEN。不加--judge不调用裁判。请求产生API用量。repeat范围1—10、concurrency范围1—4，默认均1。裁判SDK max_retries=1、timeout=30、max_tokens=2000、temperature=0；业务模型继承原配置，不增加业务重试。节点数不等同HTTP请求数，SDK内部重试不保证全部可见。

## 第一步：人工审批案例规格

案例审批与结果评审不同，都不会批准真实退货。

打开data/evaluation/cases.jsonl，逐条核对身份、输入、expect、forbid、checks和reference。checks是机器断言，expect/forbid是完整要求，未全部实现自动判断。确认规格符合你的需求后，手动将对应review_status从pending改为approved。pending也能探索运行，但不能称为人工确认的验收结果；不要因Agent答错而把金标改成错误答案。

E18/E19/E21/E22需在RAGFlow核对相关文档、原文及适用前提，把准确文档名填入relevant_documents数组。只有approved且数组非空才计算文档级Hit@5。这不代表分块适用性或回答正确性；仍需人工审查。首版真实RAG案例不运行语言裁判。测试题和答案禁止导入知识库。

修改数据后，运行指纹自动记录实际文件哈希；独立manifest为静态发布清单，修改案例后也应更新。不要拿旧哈希当成新版本。

## 第二步：查看实验并填写人工结论

每次输出到evals/results/时间-随机编号/，不覆盖旧结果：

| 文件 | 内容 |
|---|---|
| experiment.json | 代码及数据哈希、提交与脏状态、依赖和模型参数 |
| results.jsonl | 每次案例运行的结构化记录 |
| report.md / summary.json | 分组指标、失败原因、耗时、重复稳定性 |
| E编号-运行编号/result.json | 逐轮回答、状态、工具轨迹、RAG候选、裁判分数 |
| human_review.csv | 独立人工审核表 |

先看案例规格，再看逐轮answer、messages、checks.actual、state、operation、trace和events。建议首次人工评分时先不看模型分数，减少先入为主。

填写human_review.csv：reviewer填别名；verdict填pass/fail/unjudgeable，空白为未审；evidence写实际回答、状态或来源原文；reason说明理由；五个英文维度列填写0—4整数，无法判断留空。预置turn=all表示整场评分；可以另加逐轮行，但不纳入整场一致率。run_id+turn+reviewer不能重复。

五维为factual_correctness事实正确性、faithfulness知识忠实性、relevance问题相关性、actionability可操作性、expression表达质量。0明显错误/缺失；1重大缺陷；2部分满足；3基本满足且有轻微问题；4充分满足。事实与来源不可核实时不能仅因表达流畅打高分。硬规则与语言分独立，严重越权不能由均分抵消。

## 第三步：比较人工和自动化

把路径替换为本次实际实验目录：

```powershell
uv run orderagent-eval compare --result "evals/results/实际实验目录名"
```

生成comparison.json，保留人工结论，输出整场人工与机器显式硬规则的一致率、各维“人工分－裁判分”和证据。未审及无法判断不进入二元一致率；模型null不强行计算差值。人工评审范围可能比机器断言更广，一致率不是模型准确率。优先复核人工fail/机器pass及维度差值绝对值≥2的记录。

compare不重写人工表；重新运行是新实验，需要重新审核，不能沿用旧run_id。工具不会自动修改提示词、业务规则或裁判。

## 隔离、分支与指标

每案例/每次重复独立SQLite和thread_id，同案例多轮共享。受控fixture不能冒充真实RAG。on_fail默认stop，中途失败停止并记录未执行轮次；可设on_fail=continue或on_pass=stop。首版没有任意跳转与用户模拟器。

- 有效率=显式硬规则可判定通过/失败数÷计划数；中途硬规则失败仍是有效失败。
- 有效样本通过率=通过数÷有效数，另报通过数÷计划数。
- 无金标RAG为unjudgeable，目标异常为error，都保留在计划分母。
- 裁判覆盖率=完整结构评分数÷已有对话记录数；各维非null数量另报。裁判失败不改变业务结论。
- P50/P95为整案例目标耗时，含组装、检索和多轮，不含裁判；裁判耗时单独记录。小样本不能证明容量。
- 重复运行按案例列出结果分布；只有一次无法判断稳定性。

归因区分检索、规则选择、输出截断、路由/参数、状态/边界、目标异常和裁判错误。受控超时后正确转人工可通过业务检查，同时保留故障诊断。自然语言要求仍需人工/模型评审。

## 可选LangSmith上传

确认边界补充集：`uv run orderagent-eval run --cases data/evaluation/confirmation_cases.jsonl --repeat 1 --concurrency 1`。仍是受控规则夹具，不能代表真实RAG端到端通过。本地规则现场额外保存raw_content及parsing_error，不新增正文上传；已有框架追踪仍按原配置管理。

低推理对照：仅设置 `RAGFLOW_SELECTION_REASONING_EFFORT=high` 或 `low`，预算保留4096，分别运行并保存不同实验目录。不要同时改数据、提示词和模型配置后将差异全部归因于推理强度。新增 `selection_metrics` 包含成功或异常调用耗时与可用用量，summary提供选择P50/P95；样本少不作容量结论。错误放行、误转人工需人工核对和明确断言，不从泛化通过率推算。

配置LANGSMITH_API_KEY后显式执行：

```powershell
uv run orderagent-eval run --ids E01 --judge --upload
```

官方Client创建新的云端案例集，上传所选测试语句及预期。运行详情inputs/outputs/metadata隐藏，云端保留步骤、耗时、硬规则及五维反馈；完整脱敏证据与评分理由在本地。云端权限尚未实测。

不加--upload时显式关闭远程Tracing，使用官方RunCollector，即使环境LANGSMITH_TRACING=true也不自动上传。当前版本langchain-core将local视为真值，因此业务执行使用False而不是字符串local。结果目录gitignore，不存.env、授权头及隐藏推理字段。

记录代码/提示词/数据哈希、模型及裁判配置、候选原文和可用响应信息；外部知识库无快照时标记not_available，动态模型无法保证完全复现。本工具不做HTTP容量压测，也不输出真实平台退款成功率。
