# 数码电商售后 Agent

基于 **Deep Agents `create_deep_agent` + LangGraph + ChatDeepSeek** 的订单查询项目，使用 uv 管理 Python 环境与依赖。当前只实现模型接入、订单查询和命令行多轮会话。

## 安装与运行

要求 Python 3.12 或更新版本，以及 uv。在项目根目录运行：

```powershell
uv sync --locked
uv run orderagent --user-id DEMO-U-001
```

输入 `/reset` 清空会话，`/exit` 退出。

单轮查询并显示结构化工具轨迹：

```powershell
uv run orderagent --user-id DEMO-U-001 --message "查询订单 DEMO-HP-O-0001，告诉我发货状态和商品型号。" --trace
```

不调用模型，直接验证订单工具：

```powershell
uv run orderagent --user-id DEMO-U-001 --order-id DEMO-HP-O-0001
```

直接查询无需 API 密钥。当前 `DEMO-U-001` 可以查询 `DEMO-HP-O-0001`；查询其他用户的 `DEMO-HP-O-0002` 会得到与订单不存在一致的失败结果。

## 采用的框架

[agent_factory.py](src/orderagent/agents/agent_factory.py) 调用官方 `deepagents.create_deep_agent` 构建图，传入 ChatDeepSeek、订单工具、系统提示、上下文 Schema 和中间件。模型执行、工具调度与结果回传由框架完成，项目不再维护手写工具循环。

[order_agent.py](src/orderagent/agents/order_agent.py) 只适配会话：校验输入、调用图、提取回答与工具轨迹、在成功后保存最近8轮完整对话。未启用持久化 checkpointer，不跨进程保留历史。一个会话实例绑定一个可信用户；不要跨用户复用会话实例。

|依赖|职责|
|---|---|
|deepagents|`create_deep_agent` 图构建与框架中间件|
|langchain-deepseek|`ChatDeepSeek` 模型适配|
|langchain / langchain-core|`@tool`、`ToolRuntime`、消息及中间件协议|
|langgraph|图执行、运行上下文与递归限制|
|pydantic|配置、订单参数、业务结果结构校验|
|python-dotenv|加载已有 `.env`|
|pytest / ruff|开发测试、格式化及静态检查|

运行与开发依赖声明于 `pyproject.toml`，具体版本由 `uv.lock` 锁定；`.venv` 和缓存不提交。后续使用 `uv add 包名` 或 `uv add --dev 包名` 安装依赖。

## 文件命名与模块边界

统一使用 `snake_case.py`，业务模块使用单数名，并增加表达职责的后缀；类使用 `PascalCase`，函数使用 `snake_case`。`__init__.py` 只负责公开导出。

```text
src/orderagent/
├── settings.py                     配置与API基础地址规范化
├── exceptions.py                   公开执行异常
├── cli.py                          命令行入口与应用组装
├── __main__.py                     python -m orderagent 入口
├── agents/
│   ├── agent_factory.py            create_deep_agent 图工厂
│   └── order_agent.py              会话适配，不负责工具循环
├── models/
│   └── deepseek_model.py           ChatDeepSeek 模型工厂
├── prompts/
│   └── order_prompt.py             订单业务系统提示
├── tools/
│   └── order_query_tool.py         查询业务逻辑与LangChain工具适配
├── repositories/
│   └── order_repository.py         只读JSON仓储和可替换的数据访问协议
├── schemas/
│   └── order_schema.py             订单参数、上下文、结果与回复结构
└── middleware/
    └── tool_policy.py              模型工具白名单、执行拦截与参数校验

tests/
├── conftest.py
├── fake_chat_model.py
├── test_deepseek_model.py
├── test_order_query_tool.py
└── test_order_agent.py
```

旧的 `config.py`、`deepseek.py`、`orders.py`、`order_query.py` 已重命名；自定义工具注册器和 HTTP 模型客户端已删除，避免两套执行架构并存。

## 订单工具与可信上下文

订单工具在独立 `tools` 包内。`OrderQueryTool.execute` 执行可独立测试的业务逻辑，`as_langchain_tool` 转为框架 `@tool`，通过 `content_and_artifact` 同时返回模型可读内容和结构化结果。

模型参数仅包含 `order_id`。`ToolContext.user_id` 通过图调用的 `context` 注入，并由 `ToolRuntime` 在工具内部读取；不出现在模型参数 Schema 中。中间件使用 `OrderQuery` 拒绝额外字段、非法类型、空订单号和伪造身份参数。

仓储只返回归属于当前用户的订单。不存在与无权限统一返回 `ORDER_NOT_FOUND`；成功结果包括订单状态、发货状态、物流、商品名称、型号、数量和快照时间，不向模型传递整个原始档案。

`--user-id` 是本地演示者提供的上下文，不是真实身份认证。将来接入网页或平台时必须由可信登录会话提供用户身份。

## Deep Agents 默认工具的限制

`create_deep_agent(tools=[...])` 是添加业务工具，不代表自动关闭框架默认工具。本项目额外进行了以下限制：

- 文件中间件仅保留框架要求的 `read_file` 内部脚手架，使用默认内存后端；没有启用本机文件系统后端。
- 工具策略只向模型提供 `query_order`，并在执行层拦截其他工具。即使模型主动构造 `read_file`、`task` 或未知工具调用，也返回结构化拒绝结果。
- 不接入本机 `.env` 文件、终端执行或真实子代理任务；当前只有订单只读查询能力。
- 框架中间件限制单次运行最多6次模型调用、8次工具执行，图递归步数上限40。

本阶段提供同步命令行调用；未实现异步服务接口。底层图状态和中间件由框架维护，不能把其可扩展能力表述为本项目已经实现的业务功能。

## 模型配置

兼容根目录已有 `.env`，操作系统环境变量优先：

|变量|用途|
|---|---|
|`DEEP_BASE_URL`|默认 `https://api.deepseek.com`；兼容 `/v1` 以及完整 `/chat/completions` 地址，交给SDK前规范化为基础地址|
|`DEEP_API_KEY`|DeepSeek 密钥；不写入文档或业务数据|
|`DEEP-SEEK`|兼容已有的模型名称变量，默认 `deepseek-chat`|
|`DEEP_MODEL`|可选覆盖模型名称，优先于 `DEEP-SEEK`|
|`ORDER_DATA_PATH`|可选订单 JSON 路径；默认 `数据集/json/orders.json`|

模型由 `create_deepseek_model` 创建；请求超时30秒，SDK最多重试1次。真实请求可能产生供应商费用。Agent执行失败时对外返回脱敏信息，不输出供应商异常正文或内部密钥。

## 数据范围

本阶段读取 [orders.json](数据集/json/orders.json) 中的60条模拟订单。商品名称、型号和数量直接来自订单明细；订单状态是历史模拟快照，不是实时平台或物流查询。

其他 JSON 保持原样：379条历史商品主档、1140条商品事实问答、60条故障知识、2021条合成案例，以及工单、来源和演示规则。本阶段尚未使用它们实现商品咨询、故障排查或退换货。商品源目录最后更新于2016-05-09，不应当作现售商品库。

## 验证

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

当前32项测试通过。测试使用可绑定工具的假模型运行真实 `create_deep_agent` 图，验证工具执行与结构化结果、模型可见工具列表、额外身份参数拒绝、越权订单、默认工具拦截、用户上下文隔离、历史与重置、失败不提交历史、调用限制、配置与仓储校验。默认测试不调用真实供应商。

已使用当前 `.env` 完成一次真实端到端查询，确认 `ChatDeepSeek → Deep Agents → query_order → 订单结果 → 中文回答` 可运行，回答包含“待付款、未发货”和商品型号。该验证不代表完成独立效果评测。

后续可新增商品工具与数据库适配器。当前没有网页、生产登录、持久化会话、退货写入或人工通知。

## 更新日志约定

每次项目更新都在 `docs/` 下记录 Markdown 日志，文件名为 `YYYY-MM-DD+内容.md`，日期按 Asia/Shanghai 时区。

日志包含更新内容、解决的问题、涉及文件、验证结果和尚存限制；同日不同主题分别记录，同主题可追加。密钥与敏感配置不写入日志。具体协作要求见 [AGENTS.md](AGENTS.md)。

- [2026-10-21：Deep Agents 架构重构及日志规范](docs/2026-10-21+DeepAgents架构重构及日志规范.md)

## 官方资料

- [Deep Agents 自定义与 create_deep_agent](https://docs.langchain.com/oss/python/deepagents/customization)
- [Deep Agents 后端](https://docs.langchain.com/oss/python/deepagents/backends)
- [ChatDeepSeek 集成](https://docs.langchain.com/oss/python/integrations/chat/deepseek)
