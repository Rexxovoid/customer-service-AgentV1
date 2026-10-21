# OrderAgent · 数码电商售后助手

基于 **LangChain 官方 Deep Agents** 的电商售后项目，面向耳机、鼠标、键盘等数码产品，使用 DeepSeek 模型和 uv 环境。

当前版本实现 **模型接入、订单查询和命令行多轮对话**。订单来自本地模拟数据，尚未连接真实购物平台。

## 已实现的能力

|能力|实现方式|
|---|---|
|自然语言订单查询|模型识别诉求，调用 `query_order` 查询状态、物流和商品明细|
|订单归属校验|使用程序提供的用户上下文，仅返回当前用户所属订单|
|多轮会话|官方 `InMemorySaver` 按线程保存历史，支持清空会话|
|直接工具查询|绕过模型读取订单，无需 API 密钥|
|调用限制与调试|官方中间件限制模型和工具调用次数，支持原生执行更新输出|

商品咨询、故障排查、退换货、人工转接和网页界面尚未实现。

## 快速开始

### 1. 安装环境

要求 Python 3.12 或更新版本，以及 uv。在项目根目录执行：

```powershell
uv sync --locked
```

依赖按 `uv.lock` 安装到 `.venv`，无需手动激活环境。默认包含 pytest、Ruff 等开发依赖；只运行项目时可使用 `uv sync --locked --no-dev`。

### 2. 检查订单工具

以下命令不调用模型，也不消耗 API 额度：

```powershell
uv run orderagent --user-id DEMO-U-001 --order-id DEMO-HP-O-0001
```

预期返回结构化 JSON，包含 `ok`、`data`、`error`。该示例订单属于 `DEMO-U-001`，状态为待付款、未发货，商品型号为 `Djpro60w`。

### 3. 配置 DeepSeek

程序自动读取项目根目录已有的 `.env`，操作系统环境变量优先。变量用途如下：

|变量|说明|
|---|---|
|`DEEP_BASE_URL`|API 地址，默认 `https://api.deepseek.com`；兼容 `/v1` 或完整 `/chat/completions` 地址|
|`DEEP_API_KEY`|API 密钥，自然语言查询必需|
|`DEEP-SEEK`|模型名称，沿用已有变量名；默认 `deepseek-chat`|
|`DEEP_MODEL`|可选模型名称，优先于 `DEEP-SEEK`|
|`ORDER_DATA_PATH`|可选订单文件路径，默认 `data/orders.json`；自定义相对路径按运行目录解析|

密钥不写入源码、日志或数据文件。真实模型调用可能产生供应商费用。

### 4. 开始对话

交互模式：

```powershell
uv run orderagent --user-id DEMO-U-001
```

单轮查询：

```powershell
uv run orderagent --user-id DEMO-U-001 --message "查询订单 DEMO-HP-O-0001，告诉我发货状态和商品型号。"
```

显示框架原生执行更新及工具消息：

```powershell
uv run orderagent --user-id DEMO-U-001 --message "查一下 DEMO-HP-O-0001" --trace
```

常用参数和交互命令：

|参数或命令|用途|
|---|---|
|`--user-id`|必填，设置本次命令行演示用户|
|`--message`|进行一次自然语言查询；省略则进入交互模式|
|`--order-id`|直接查询订单，不能与 `--message` 同时使用|
|`--trace`|显示官方原生执行更新，不是单个 JSON 回复对象|
|`/reset`|删除当前线程检查点，开始新会话|
|`/exit`|退出交互模式|

使用 `uv run orderagent --help` 查看帮助。消息长度限制为1—4000字符。

## 架构与模块分工

```text
用户 → CLI → create_deep_agent / LangGraph ↔ ChatDeepSeek
                         ↓ 工具调用
                     query_order
                         ↓ 归属校验
                JsonOrderRepository → data/orders.json
```

[图工厂](src/orderagent/agents/agent_factory.py)使用官方 `create_deep_agent` 组装模型、工具和中间件；[命令行入口](src/orderagent/cli.py)直接调用 `graph.invoke`，不维护平行的工具执行框架。

- **模型**：官方 `ChatDeepSeek`，默认请求超时30秒，SDK最多重试1次。
- **工具**：独立 `tools` 包，通过官方 `@tool` 和 `ToolRuntime` 接入；结构化结果使用 `content_and_artifact` 返回。
- **会话**：官方 `InMemorySaver` 与 `thread_id`，仅在当前进程内保留，不跨进程恢复。
- **限制**：默认每次运行最多6次模型调用、8次工具执行，图递归步数上限40。
- **消息**：使用官方消息结构；`--trace` 通过 `invoke(print_mode="updates")` 展示调试信息。尚未提供逐字输出参数。

## 目录结构

```text
orderagent/
├── pyproject.toml                 依赖、命令入口和开发工具配置
├── uv.lock                        依赖锁文件
├── .python-version                Python版本
├── AGENTS.md                      项目协作约定
├── README.md                      项目说明
├── src/orderagent/
│   ├── cli.py                     命令行入口
│   ├── settings.py                环境配置
│   ├── __main__.py                模块启动入口
│   ├── agents/agent_factory.py    官方Agent图工厂
│   ├── models/deepseek_model.py   模型配置
│   ├── tools/order_query_tool.py  订单查询工具
│   ├── repositories/order_repository.py  JSON仓储
│   ├── schemas/order_schema.py    参数、上下文和结果结构
│   ├── prompts/order_prompt.py    系统提示词
│   └── middleware/tool_policy.py  工具和参数限制
├── tests/                         配置、工具、官方图和CLI测试
├── data/                          独立JSON数据集合、来源和清单
└── docs/                          按日期和主题保存的更新日志
```

本地 `.env`、`.venv` 和可再生缓存由 Git 忽略。包内 `__init__.py` 用于公开导出。

## 数据说明

|文件|记录数|当前用途|
|---|---:|---|
|[orders.json](data/orders.json)|60|订单查询运行数据|
|[products.json](data/products.json)|379|后续商品咨询资料|
|[product_faqs.json](data/product_faqs.json)|1140|后续商品事实问答资料|
|[support_knowledge.json](data/support_knowledge.json)|60|后续故障排查资料|
|[after_sales_cases.json](data/after_sales_cases.json)|2021|模拟售后开发案例|
|[tickets.json](data/tickets.json)|18|后续内部工单样例|

商品、问答、故障知识和模拟案例合计3600条，模拟订单和工单另计。事实问答与商品主档共享证据，模拟案例共享场景模板，不应将总条数当作独立商品数或真实用户规模。

`data/` 另保留来源登记、演示规则、元数据、历史转换报告、来源许可和文件哈希清单。每个集合只保留一个独立文件，当前代码仅读取订单集合。

订单字段中，`id` 为订单号，`user_id` 为所属用户，`items` 为商品明细，`shipping_status` 为发货状态，`logistics_status` 为物流状态，`snapshot_at` 为快照时间。

**数据边界**：订单和工单均为模拟记录；商品资料来自2016-05-09更新的历史开放目录，不代表当前在售、现价、库存或售后政策。原始英文参数未逐条翻译。

## 查询边界与会话行为

工具参数只允许订单号。用户身份从程序运行上下文注入，不允许模型通过额外参数切换用户；订单不存在与不属于当前用户统一返回 `ORDER_NOT_FOUND`。

`--user-id` 是本地演示上下文，不是真实身份认证。未来接入服务时，必须从可信登录会话获得用户身份，并保证不同用户使用不同线程。

现有工具策略只开放 `query_order`，拦截其他工具和非法参数。未启用本机文件系统后端；模型不能借助框架文件工具读取本地 `.env`。

检查点可能保留失败运行的部分状态，必要时用 `/reset` 清理。当前没有自写历史回滚或固定轮数截断；长会话总结效果尚未独立评测。

## 测试与检查

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

最近一次代码验证记录为 **34项测试通过**，覆盖订单归属、参数校验、工具限制、检查点历史与删除、线程隔离、调用次数、CLI错误脱敏、配置和仓储。测试使用假模型，不调用真实供应商。

此前已验证真实API订单查询流程。本次README重写仅核对源码、锁文件、安装版本、数据数量和文档链接，未重新运行测试或真实API调用。详细验证记录见 [更新日志](docs/)。

当前锁定并安装的核心版本：

|依赖|版本|
|---|---|
|deepagents|0.7.29|
|langchain|1.4.6|
|langchain-core|1.6.12|
|langgraph|1.2.16|
|langchain-deepseek|1.1.1|

## 开发约定与后续计划

在现有架构内优先最小修改。使用官方API前核对当前依赖版本；工具调用、执行、消息、重试和流式输出优先采用官方封装。确需自写模块时，先说明必要性与改动计划，等待确认后实现。完整要求见 [AGENTS.md](AGENTS.md)。

每次更新在 `docs/` 写日志，文件名为 `YYYY-MM-DD+内容.md`，包含更新内容、解决的问题、涉及文件、验证结果及限制，日期使用 Asia/Shanghai 时区。

后续功能候选为商品咨询、故障排查、退换意向与人工交接，具体实现另行讨论。当前没有网页、生产登录、真实电商接口、退货写入或人工通知。

## 官方参考

- [Deep Agents 自定义](https://docs.langchain.com/oss/python/deepagents/customization)
- [Deep Agents 后端](https://docs.langchain.com/oss/python/deepagents/backends)
- [ChatDeepSeek 集成](https://docs.langchain.com/oss/python/integrations/chat/deepseek)
