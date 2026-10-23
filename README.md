# OrderAgent · 数码电商售后助手

基于 **LangChain 官方 Deep Agents** 的电商售后项目，面向耳机、鼠标、键盘等数码产品，使用 DeepSeek 模型和 uv 环境。

当前版本实现 **订单查询、商品参数查询、质量问题基础排查、模拟人工交接和命令行多轮对话**，使用DeepSeek模型。订单与客服交接均为演示，尚未连接真实购物平台或真人坐席。

## 已实现的能力

|能力|实现方式|
|---|---|
|自然语言订单查询|模型识别诉求，调用 `query_order` 查询状态、物流和商品明细|
|订单归属校验|使用程序提供的用户上下文，仅返回当前用户所属订单|
|商品信息查询|调用 `query_products` 按品类、品牌、型号、名称或ID查询耳机、鼠标、键盘参数|
|候选型号确认|多条匹配最多展示5款及已有参数，请用户确认型号；未记载的规格说明未知|
|质量问题回答|调用 `query_quality_support`，按品类、故障及适用条件查询知识，最多给出一次自动方案|
|人工交接接口|无适用知识或方案未解决时交接；可替换平台适配接口，当前模拟成功工号001或转接错误|
|多轮会话|官方 `InMemorySaver` 按线程保存历史，支持清空会话|
|直接工具查询|绕过模型读取订单，无需 API 密钥|
|调用限制与调试|官方中间件限制模型和工具调用次数，支持原生执行更新输出|

退换货、真实平台客服接入和网页界面尚未实现。

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
|`PRODUCT_DATA_PATH`|可选商品文件路径，默认 `data/products.json`；自定义相对路径按运行目录解析|
|`SUPPORT_DATA_PATH`|可选故障知识文件路径，默认 `data/support_knowledge.json`|
|`MOCK_HANDOFF_SCENARIO`|模拟客服场景，`success`（默认）或 `error`；只由程序配置控制|

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

商品性能查询：

```powershell
uv run orderagent --user-id DEMO-U-001 --message "介绍罗技 M185 鼠标的性能，包括DPI、连接方式、按键和续航。"
uv run orderagent --user-id DEMO-U-001 --message "鼠标有哪些型号及已知参数？"
```

只提供品类时，先展示候选，再在交互模式中确认型号。商品查询使用公开资料，不需要订单号；CLI仍要求演示用户参数。

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

### 5. 测试质量问题与模拟客服

启动交互模式，在同一次进程中依次输入「我的耳机左耳没有声音了」和「没有解决」。首轮只给一次有知识依据的基础检查，第二轮直接调用人工接口并返回固定结果：

```text
【模拟客服】对接人工客服成功，由工号001为你服务。
```

测试失败场景，在PowerShell中设置环境变量并重新启动：

```powershell
$env:MOCK_HANDOFF_SCENARIO = "error"
uv run orderagent --user-id DEMO-U-001
```

重复同样对话，第二轮应返回「【模拟客服】转人工错误，请稍后重试或联系平台客服。」，不会继续排查或谎称成功。改为 `success` 后重启可恢复成功场景；也可用 `Remove-Item Env:MOCK_HANDOFF_SCENARIO` 清除临时覆盖，恢复 `.env` 或默认配置。

直接输入「转人工」也能测试接口。无匹配问题会直接交接，不编造方案；例如「耳机出现未知故障XYZ」。每次单独运行 `--message` 都是新会话，要测试“未解决”请使用交互模式。

## 架构与模块分工

```text
用户 → CLI → create_deep_agent / LangGraph ↔ ChatDeepSeek
                         ↓ 工具调用
                     ├─ query_order → 归属校验
                     │    └─ JsonOrderRepository → data/orders.json
                     ├─ query_products → 型号匹配／候选确认
                     │    └─ JsonProductRepository → data/products.json
                     ├─ query_quality_support → 一次方案或自动交接
                     │    └─ JsonSupportRepository → data/support_knowledge.json
                     └─ handoff_to_human → HandoffGateway → MockHandoffGateway
```

[图工厂](src/orderagent/agents/agent_factory.py)使用官方 `create_deep_agent` 组装模型、工具和中间件；[命令行入口](src/orderagent/cli.py)直接调用 `graph.invoke`，不维护平行的工具执行框架。

订单工具位于 [order_query_tool.py](src/orderagent/tools/order_query_tool.py)：普通函数 `query_order(repository, arguments, context)` 负责查询并整理结果，供CLI直接查询与Agent共同复用；`create_order_query_tool(repository)` 绑定仓储并返回官方 `@tool`，无需实例化工具类。仓储启动时将JSON订单列表建立为订单号字典，查询时通过 `dict.get(order_id)` 精确取值，再校验用户归属。

- **模型**：官方 `ChatDeepSeek`，默认请求超时30秒，SDK最多重试1次。
- **工具**：独立 `tools` 包，通过官方 `@tool` 和 `ToolRuntime` 接入；订单和商品使用 `content_and_artifact`，质量与交接使用官方 `Command(update=...)` 更新状态并添加带artifact的 `ToolMessage`。
- **会话**：官方 `InMemorySaver` 与 `thread_id`，仅在当前进程内保留，不跨进程恢复。
- **限制**：默认每次运行最多6次模型调用、8次工具执行，图递归步数上限40。
- **消息**：使用官方消息结构；`--trace` 通过 `invoke(print_mode="updates")` 展示调试信息。尚未提供逐字输出参数。

质量阶段使用扩展的官方 `DeepAgentState` 和原检查点保存，不创建另一套会话存储。常见明确反馈如「没有解决」「还是没声音」由现有工具策略中间件的官方 `before_model` 钩子直接交接并结束本轮；其他表述由模型识别并调用交接工具。质量方案和交接结果由程序输出，避免工具执行后再生成其他排查方案。一次会话最多一次自动方案，新独立故障使用 `/reset` 开启新会话。

人工平台接口位于 [integrations/handoff.py](src/orderagent/integrations/handoff.py)：`HandoffGateway.transfer(HandoffRequest) -> HandoffResponse`。请求包含稳定 `request_id`、可信用户身份、问题、原因和已尝试方案；响应包含连接状态、是否模拟和客服工号。将新平台适配对象通过 `create_order_agent(settings, handoff_gateway=adapter)` 注入即可替换模拟实现。这是Python适配接口，目前没有HTTP服务、真实聊天通道或人工通知。

接口失败、异常或响应不合法均不能返回成功；成功和失败结果都保存在当前会话状态，重复请求不重复调用接口、不静默重试。未来真实适配器需按 `request_id` 保证幂等并自行设置网络超时。状态目前仅进程内保存，不具有生产级持久化保证。

## 提示词配置

订单、商品和质量问题处理提示词保存在 [order_prompt.yaml](src/orderagent/prompts/order_prompt.yaml)，使用 YAML 多行字符串，键名为 `order_system_prompt`，沿用现有导出变量以保持兼容。

[order_prompt.py](src/orderagent/prompts/order_prompt.py)通过 Python 标准库 `importlib.resources` 读取包内资源，用 PyYAML 的 `yaml.safe_load` 加载后导出 `ORDER_SYSTEM_PROMPT`。需要提示词的函数继续按以下方式导入：

```python
from orderagent.prompts.order_prompt import ORDER_SYSTEM_PROMPT
```

图工厂将该变量传给官方 `create_deep_agent(system_prompt=...)`。编辑提示词时修改 YAML 正文即可，保留键名和缩进；重新启动进程后生效。当前没有模板渲染、热加载或自定义提示词管理框架。

PyYAML使用项目原已锁定的6.0.3版本，并已声明为直接依赖；提示词文件随wheel安装包一起分发，不依赖终端当前工作目录。

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
│   ├── tools/product_query_tool.py 商品参数查询工具
│   ├── tools/support_tools.py      质量与交接工具工厂
│   ├── integrations/handoff.py     人工平台接口与模拟实现
│   ├── repositories/order_repository.py  JSON仓储
│   ├── repositories/product_repository.py 商品只读匹配
│   ├── repositories/support_repository.py 故障知识匹配
│   ├── schemas/order_schema.py    参数、上下文和结果结构
│   ├── schemas/product_schema.py  商品参数及证据结构
│   ├── schemas/support_schema.py  质量参数、知识和会话阶段
│   ├── prompts/order_prompt.yaml 系统提示词正文
│   ├── prompts/order_prompt.py   YAML加载与变量导出
│   └── middleware/tool_policy.py  工具和参数限制
├── tests/                         配置、工具、官方图和CLI测试
├── data/                          独立JSON数据集合、来源和清单
└── docs/                          按日期和主题保存的更新日志
```

本地 `.env`、`.venv` 和可再生缓存由 Git 忽略。包内 `__init__.py` 用于公开导出。

## 数据说明

|文件|记录数|当前用途|
|---|---:|---|
|[orders.json](data/orders.json)|7260|原60条模拟订单与新增7200条测试订单|
|[products.json](data/products.json)|379|商品查询运行数据：鼠标90、键盘89、耳机200|
|[product_faqs.json](data/product_faqs.json)|1140|后续商品事实问答资料|
|[support_knowledge.json](data/support_knowledge.json)|68|统一故障知识：原60条摘要与新增8条案例；其中3条为明确标注的项目模拟指导|
|[after_sales_cases.json](data/after_sales_cases.json)|2021|模拟售后开发案例|
|[tickets.json](data/tickets.json)|1518|原18条模拟工单与新增1500条关联测试工单；尚未接入运行工单服务|

商品、问答、故障知识和模拟案例合计3608条，模拟订单和工单另计。新增8条质量案例已并入故障知识，保留原记录ID、来源性质和适用条件。事实问答与商品主档共享证据，模拟案例共享场景模板，不应将总条数当作独立商品数或真实用户规模。

2026-10-23新增测试数据按每类3400条组织：2400条订单、500条工单、500条Markdown问答，三类共10200条。结构化数据追加到既有JSON；另外1500条问答保存在 [ragflow_knowledge](data/ragflow_knowledge)，包含180条商品事实问答、60条条件式故障问答和1260条售后规则场景。17份Markdown中有2份基础参考文档，不计入问答数量。

售后规则场景基于20个主题×21种条件构造，区分官方法规摘要与项目流程，不是1260条独立法规。法律适用范围、商品时效、模拟标记及来源均在正文中保留；没有制造新的商品参数、真实交易或品牌保修承诺。导入方式、数量口径和订单查询示例见 [数据使用说明](data/README.md)。本次只生成文件，未上传或接入RAGFlow；问答不能同时作为独立评测集。

新增订单查询示例：

```powershell
uv run orderagent --user-id LOAD-HP-U-0001 --order-id LOAD-HP-O-000001
```

新增用户和订单按品类隔离，每类120个模拟用户。使用其他用户（包括 `DEMO-U-001`）查询上述订单应返回 `ORDER_NOT_FOUND`。订单金额及物流均为模拟，不能当作实际价格或平台执行结果。

`data/` 另保留来源登记、演示规则、元数据、历史转换报告、来源许可和文件哈希清单。当前运行读取订单、商品和统一故障知识集合；不把模拟售后案例或旧工单当作已验证的维修方案。新增案例来源自包含在各记录中。原 `support_cases.json` 已合并删除，知识文件路径统一通过 `SUPPORT_DATA_PATH` 配置，不再读取 `SUPPORT_CASES_PATH`。

订单字段中，`id` 为订单号，`user_id` 为所属用户，`items` 为商品明细，`shipping_status` 为发货状态，`logistics_status` 为物流状态，`snapshot_at` 为快照时间。

**数据边界**：订单和工单均为模拟记录；商品资料来自2016-05-09更新的历史开放目录，不代表当前在售、现价、库存或售后政策。原始英文参数未逐条翻译。

## 查询边界与会话行为

订单工具参数只允许订单号。用户身份从程序运行上下文注入，不允许模型通过额外参数切换用户；订单不存在与不属于当前用户统一返回 `ORDER_NOT_FOUND`。

商品工具只接受 `query`，精确ID、型号和源SKU优先，其后使用有限的中英文品类、品牌别名与关键词匹配。模型从问题中提取商品关键词；尚不提供模糊纠错、语义检索或推荐排序。详情依据原始 `description` 和 `evidence_groups`，保留来源与更新时间；空证据分组不能推断为不支持。当前售价、库存以及资料未记载的DPI、回报率、续航均不能补全。

`--user-id` 是本地演示上下文，不是真实身份认证。未来接入服务时，必须从可信登录会话获得用户身份，并保证不同用户使用不同线程。

现有 `BusinessToolPolicyMiddleware` 只开放 `query_order`、`query_products`、`query_quality_support` 和 `handoff_to_human`，分别校验参数。涉及质量或交接的并行工具批次会被拒绝，防止多份方案和阶段写入冲突。未启用本机文件系统后端；模型不能借助框架文件工具读取本地 `.env`。

故障匹配采用有限症状别名，不是语义搜索；有系统、品牌限制的方案需用户提供相应信息，未知时不推测。通用案例明确标注为项目模拟指导。自然语言反馈识别未做全面准确率评估，任意否定、引用或复杂多意图句仍可能误判；模型须遵循提示词补充识别未解决并交接。禁止第二次自动知识方案由程序状态控制。

检查点可能保留失败运行的部分状态，必要时用 `/reset` 清理。当前没有自写历史回滚或固定轮数截断；长会话总结效果尚未独立评测。

## 测试与检查

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

最近一次代码验证记录为 **99项测试通过**，包含订单与商品回归、质量方案、未解决后交接、无匹配、接口失败与异常、不合法响应、幂等、适用条件、并行拦截、会话隔离与重置及配置。自动化测试使用假模型，不调用真实供应商；Ruff检查及格式检查通过。

此前真实DeepSeek API已验证耳机左耳无声查询及未解决后交接，在模拟成功、失败配置下均符合预期；详见[质量与模拟人工更新日志](docs/2026-10-22+质量问题回答与模拟人工交接.md)。本次数据扩充未重复真实模型调用，未执行服务器压测或RAGFlow解析评测。

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

后续功能候选为真实客服平台对接、退换意向与持久化，具体实现另行讨论。当前没有网页、生产登录、真实电商接口、退货写入或人工通知。

## 官方参考

- [Deep Agents 自定义](https://docs.langchain.com/oss/python/deepagents/customization)
- [Deep Agents 后端](https://docs.langchain.com/oss/python/deepagents/backends)
- [ChatDeepSeek 集成](https://docs.langchain.com/oss/python/integrations/chat/deepseek)
