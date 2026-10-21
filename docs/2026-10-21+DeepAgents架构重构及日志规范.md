# Deep Agents 架构重构及更新日志规范

- 日期：2026-10-21（Asia/Shanghai）
- 记录性质：补记当天已完成的架构重构，并记录本次新增的日志约定。
- 范围：模型接入、订单查询、文件命名及项目文档。

## 更新内容

1. 使用 `deepagents.create_deep_agent` 构建执行图，由 Deep Agents／LangGraph 完成模型与工具调用循环，移除原先的手写循环。
2. 使用 `langchain_deepseek.ChatDeepSeek` 接入模型，兼容现有 `DEEP_BASE_URL`、`DEEP_API_KEY`、`DEEP-SEEK` 配置名称。
3. 保持订单工具独立，通过 `@tool`、`ToolRuntime` 和结构化结果接入框架。
4. 统一业务文件为单数 `snake_case` 命名，使用职责后缀区分模型、工具、仓储、结构和提示词。
5. 增加工具策略中间件，只向模型开放 `query_order`，执行时拒绝其他工具和额外身份参数。
6. 通过 uv 更新依赖及锁文件，更新架构、使用方式和限制说明。
7. 本次新增项目根目录 `AGENTS.md` 的协作约定：后续更新均在 `docs/` 记录日期、更新内容和解决的问题；README 同步增加说明。

## 解决的问题

|问题|处理方式|
|---|---|
|原实现未采用用户要求的 Deep Agents 架构|引入真实 `create_deep_agent` 执行图，移除自定义工具调度器及 HTTP 模型客户端|
|文件名称过于笼统，职责不易区分|统一为 `settings.py`、`deepseek_model.py`、`order_query_tool.py`、`order_repository.py`、`order_schema.py` 等|
|框架默认工具超出当前订单查询范围|通过工具策略过滤并拦截文件、子代理及其他非订单调用；不启用本机文件系统后端|
|模型参数可能携带其他用户身份|可信用户身份仅从运行上下文注入，工具参数额外字段校验禁止传入 `user_id`|
|工具运行时注入与显式参数 Schema 不兼容|由框架推导包含注入信息的工具结构，模型侧仅展示订单号；执行前仍用业务 Schema 严格校验|
|重试或多轮查询可能混入旧工具轨迹|会话适配层区分既有与新增工具消息，成功后保存最近8轮完整对话|
|更新缺少持续可查的记录|建立 `docs/YYYY-MM-DD+内容.md` 约定，并在项目协作文件与 README 中固定说明|

## 涉及文件

- `pyproject.toml`、`uv.lock`：框架依赖与版本锁定。
- `src/orderagent/settings.py`：配置与模型基础地址规范化。
- `src/orderagent/models/deepseek_model.py`：模型工厂。
- `src/orderagent/agents/agent_factory.py`：Deep Agent 图工厂。
- `src/orderagent/agents/order_agent.py`：会话适配。
- `src/orderagent/tools/order_query_tool.py`：订单工具与框架适配。
- `src/orderagent/repositories/order_repository.py`：只读订单仓储。
- `src/orderagent/schemas/order_schema.py`：结构化参数、上下文及返回结果。
- `src/orderagent/middleware/tool_policy.py`：工具白名单与参数校验。
- `src/orderagent/prompts/order_prompt.py`：业务提示词。
- `tests/`：框架、模型配置、订单工具及会话行为测试。
- `AGENTS.md`、`README.md`、本日志：本次文档约定更新。

## 验证结果

以下为前一次架构重构完成时已执行的验证，本次仅更新文档，没有重复运行代码测试：

- `uv sync --locked`：通过。
- `pytest`：32 项测试通过。
- Ruff 静态检查与格式检查：通过。
- 直接订单工具查询：返回模拟订单 `DEMO-HP-O-0001` 的结构化结果。
- 真实 API 查询：完成 `ChatDeepSeek → Deep Agents → query_order → 中文回答`，确认返回待付款、未发货及商品型号。

本次检查：确认日志位于 `docs/`，文件名遵循约定，包含更新内容、解决的问题、涉及文件、验证结果及限制。

## 尚存限制

- 当前只有模型接入和订单查询，没有实现商品咨询、故障排查、退换货、网页或人工通知。
- 订单来自本地模拟 JSON，查询结果为快照，不是实时电商平台数据。
- 命令行用户上下文用于演示，不等同于正式登录认证。
- 单次真实调用仅验证连通性与流程，不代表完成独立效果评测。
