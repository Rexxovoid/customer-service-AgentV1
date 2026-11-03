# OrderAgent · 数码电商售后助手

基于 LangChain 官方 **Deep Agents / LangGraph**，使用 DeepSeek、RAGFlow 和 uv，面向耳机、鼠标、键盘，支持本地网页聊天和命令行对话。

当前支持订单查询、商品参数查询、质量排查、退货规则咨询、模拟申请入口和模拟人工交接。**尚未接入真实购物平台、真实退款、真人坐席或生产登录认证。**

## 快速开始

要求 Python 3.12+ 和 uv，在项目根目录执行：

```powershell
uv sync --locked

# 直接查询本人订单：不调用模型，无需模型密钥。
uv run orderagent --user-id DEMO-U-001 --order-id DEMO-HP-O-0001

# 配置.env后开始多轮对话。
uv run orderagent --user-id DEMO-U-004
```

`--user-id`是本地演示身份，不是真实登录认证。请使用与订单归属一致的用户；订单号来自[data/orders.json](data/orders.json)。

### 浏览器聊天

```powershell
uv run orderagent-web --user-id DEMO-U-004
```

打开 **http://127.0.0.1:8000**，输入问题后点击发送，或按Enter发送、Shift+Enter换行。端口被占用可加`--port 8001`。停止服务在终端按Ctrl+C。

页面支持连续问答、快捷问题、等待与错误提示及新建对话；刷新可读取当前进程中的聊天历史。新建对话清空当前聊天和咨询计数，保留售后台账。服务重启后聊天历史不保留。演示身份由启动命令固定，页面不能切换用户；不同浏览器会话独立，同一浏览器的多个标签页共享会话。

首期完整回答返回后展示，不是主Agent逐字流式输出。人工与退货继续使用模拟接口；无需安装Node或单独启动前端。默认仅本机访问，无生产认证与公网部署。Web接口及限制见[WEB规则](docs/rules.md#本地网页聊天)。

网页错误会显示“问题编号”。在`runtime/logs/requests.jsonl`查HTTP状态、耗时与异常分类，在`runtime/logs/agent.jsonl`用相同`request_id`查模型/工具阶段、工具结果码及RAG诊断。两类文件均自动轮转，默认不保存聊天和证据正文。HTTP 200不等于回答正确；网络断开且未收到响应时，页面可能拿不到编号。

质量咨询只展示整理后的步骤和必要提醒，不向用户附加知识库原文、文档名和分块ID。来源标识可在Agent日志中定位；需完整原文比对时使用现有`RAGFLOW_EVIDENCE_DEBUG`受控调试（默认关闭）。

```powershell
rg --no-ignore --fixed-strings "页面显示的问题编号" runtime/logs
```

```powershell
uv run orderagent --user-id DEMO-U-004 --message "订单 DEMO-HP-O-0004 的发货状态是什么？"
uv run orderagent --user-id DEMO-U-004 --message "介绍罗技 M185 鼠标的已有参数"
uv run orderagent --user-id DEMO-U-004 --trace
```

交互命令：`/reset`清空当前内存会话，`/exit`退出。重置不会删除持久退货记录。`--message`是单轮；`--order-id`绕过模型，与`--message`互斥。更多参数见`uv run orderagent --help`。

## 配置

根目录`.env`自动加载，系统环境变量优先；不提交密钥。以下是常用项，完整配置以[Settings](src/orderagent/settings.py)为准。

| 配置 | 用途 |
|---|---|
| `DEEP_API_KEY` / `DEEP_BASE_URL` | 模型密钥和服务地址；默认官方DeepSeek地址 |
| `DEEP_MODEL` / `DEEP-SEEK` | 模型名，前者优先；保留已有变量名 |
| `RAGFLOW_API_URL` / `RAGFLOW_API_KEY` | 配置远程知识库；未配置URL时使用本地知识，已启用远程后异常不静默降级 |
| `RAGFLOW_DATASET_NAME` / `RAGFLOW_ASSISTANT_NAME` | 需唯一匹配，助手须绑定目标知识库 |
| `RAGFLOW_SELECTION_REASONING_EFFORT` | 规则选择思考强度，默认`low`，思考保持开启；不改变主Agent配置 |
| `RAGFLOW_SELECTION_MAX_TOKENS` | 规则选择预算，默认4096；低强度不保证稳定提速 |
| `RAGFLOW_TIMEOUT_SECONDS` | RAG请求超时，默认60秒，不是整轮总时限 |
| `AFTER_SALES_DB_PATH` | 默认`runtime/after_sales.sqlite3`，持久保存退货操作与幂等信息 |
| `MOCK_HANDOFF_SCENARIO` | 模拟人工成功或失败：`success` / `error` |
| `RETURN_SELF_SERVICE_ENTRY` | 可配置的后台退货入口说明 |

模型及裁判调用产生供应商用量。规则选择低强度参数目前仅核对了官方DeepSeek服务；服务和动态模型别名变化会影响复现。

## 使用流程

| 诉求 | 当前行为 |
|---|---|
| 订单查询 | 精确订单号检索、校验归属，返回本地快照 |
| 商品查询 | 按品类/品牌/型号/ID检索，多候选先确认，未知参数不编造 |
| 质量问题 | 按来源及条件提供排查；可多轮咨询，不登记为已售后 |
| 明确退货 | 独立收集订单、目标商品及必要条件；不强制先排查 |
| 仅问退货条件 | 解释规则，不视为申请或执行确认 |
| 无可靠依据或明确人工请求 | 按规则转模拟人工，成功/失败如实展示 |

质量与退货按当前问题分别计数，第4—5次建议人工，第6次自动交接；补信息、确认和进度查询不计次。详细触发条件、强制层级与限制统一见[当前边界规范](docs/rules.md)，此处不复制全部策略。

退货在规则支持后进入待确认，指令单独一行：

```text
确认退货 DEMO-HP-O-0004
```

允许首尾空白及末尾一个`。 . ！ !`，拒绝重复标点、否定引用及混杂内容。已有待确认申请且载荷一致时，由官方中间件短路主模型请求，仍经现有工具核对身份、绑定、幂等和规则；自然语言与内容变更仍走模型。当前只生成**模拟入口**，不代表提交、批准或退款。有效入口复用，处理中/结果不明/可信终态由持久台账保护。

## 目录与职责

```text
src/orderagent/
  agents/          官方create_deep_agent组装
  models/          官方ChatDeepSeek配置
  tools/           业务工具、问题计数和确认校验
  schemas/         参数、消息数据与状态字段
  middleware/      执行钩子、确定性短句与工具回复收口
  integrations/    RAGFlow、模拟退货和人工适配器
  repositories/    JSON资料与SQLite台账
  prompts/         YAML提示词及加载
  evaluation/      独立测评入口、检查与报告
  web.py           本地Web入口与浏览器会话接入
  static/          原生聊天页面、样式与交互
data/              业务JSON、RAG导入材料、独立测评集
evals/             测评指南，results/与debug/为本地产物
docs/              当前规范、规划、维护说明和日期日志
  archive/plans/   历史实施方案，不作为当前边界
runtime/           日常持久数据，不与测评数据库混用
```

执行链路：CLI或本地网页 → 官方Deep Agent → 工具 → 仓储/知识库/模拟接口。会话检查点在进程内，退货幂等信息在SQLite；没有另建执行框架。入口见[agent_factory.py](src/orderagent/agents/agent_factory.py)。

## 测试与测评

```powershell
uv run pytest -q
uv run ruff check src tests
uv run orderagent-eval validate
uv run orderagent-eval run --ids E05,E06,E13,E15 --repeat 3 --concurrency 2
uv run orderagent-eval run --cases data/evaluation/confirmation_cases.jsonl --repeat 3 --concurrency 2
```

主集20条，优化补充8条、确认补充2条分开执行。案例审批与结果人工评审是两件事；机器通过仅表示显式断言通过，受控规则不等于真实RAG验证。完整命令、裁判配置、指标与人工对比见[测评操作指南](evals/EVALUATION_GUIDE.md)。

诊断正文默认不落盘，需要时在当前PowerShell设置后重启：

```powershell
$env:RAGFLOW_EVIDENCE_DEBUG = "true"
uv run orderagent --user-id DEMO-U-004 --trace
```

记录位于`evals/debug/rag_evidence/`，含实际输入、可见原始输出、解析详情和规范化匹配结果，可能包含业务信息。目录Git忽略；最多20份后停止新增，需手动清理。排查完设为`false`并重启。`--trace`原始输出不等于已脱敏日志。更多诊断边界见[rules.md](docs/rules.md)。

## 文档导航

- [文档索引](docs/index.md)：区分当前规范、操作指南、规划和历史。
- [边界维护分工](docs/boundary-maintenance.md)：修改边界时应该检查哪些层和测试。
- [当前边界规范](docs/rules.md)：当前实现、代码强制、提示词约束与未实现能力。
- [测评操作指南](evals/EVALUATION_GUIDE.md)：运行、案例审批、人工评分和自动对比。
- [数据说明](data/README.md) / [测评数据说明](data/evaluation/README.md)：数据来源与独立管理。
- [未来目标](docs/maintask.md)：后续网页、工作台、知识管理、部署与性能规划。

历史方案和日期日志用于追踪演进，不替代当前规则；每次更新按[AGENTS.md](AGENTS.md)维护对应规范和日期日志。
