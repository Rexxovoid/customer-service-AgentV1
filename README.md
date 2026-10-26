# OrderAgent · 数码电商售后助手

基于 **LangChain 官方 Deep Agents** 的电商售后项目，面向耳机、鼠标、键盘等数码产品，使用 DeepSeek 模型和 uv 环境。

当前版本实现 **订单查询、商品参数查询、质量问题基础排查、退货引导与模拟申请入口、模拟人工交接和命令行多轮对话**，使用DeepSeek模型。订单、退货平台与客服交接均为演示，尚未连接真实购物平台或真人坐席。

## 已实现的能力

|能力|实现方式|
|---|---|
|自然语言订单查询|模型识别诉求，调用 `query_order` 查询状态、物流和商品明细|
|订单归属校验|使用程序提供的用户上下文，仅返回当前用户所属订单|
|商品信息查询|调用 `query_products` 按品类、品牌、型号、名称或ID查询耳机、鼠标、键盘参数|
|候选型号确认|多条匹配最多展示5款及已有参数，请用户确认型号；未记载的规格说明未知|
|质量问题回答|调用 `query_quality_support`，按品类、故障及适用条件查询知识，最多给出一次自动方案|
|人工交接接口|无适用知识或方案未解决时交接；可替换平台适配接口，当前模拟成功工号001或转接错误|
|退货引导|收集订单号和原因，核对归属，按用户描述匹配RAG规则，确认后幂等提供模拟入口；累计第4次或明确人工时交接，RAG无适用方案/异常立即交接|
|RAGFlow知识检索|官方SDK检索指定知识库，核对助手绑定，结构化选择原文；质量或退货无适用依据/服务异常均立即转人工|
|售后状态|订单初始未售后，绑定本人订单后记录实际处理结果，SQLite持久化并在订单查询时返回|
|多轮会话|官方 `InMemorySaver` 按线程保存历史，支持清空会话|
|直接工具查询|绕过模型读取订单，无需 API 密钥|
|调用限制与调试|官方中间件限制模型和工具调用次数，支持原生执行更新输出|

实际提交退货、退换审批、退款、真实平台客服接入和网页界面尚未实现。

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
|`MOCK_RETURN_SCENARIO`|兼容旧值：unknown/eligible均模拟入口成功；ineligible返回拒绝，error抛错；不再控制质量资格|
|`RETURN_SELF_SERVICE_ENTRY`|自助入口说明；默认示例为“我的订单 → 订单详情 → 退货/退款”，实际按钮以平台为准|
|`RAGFLOW_API_URL`|RAGFlow服务根地址，配置后启用远程检索，兼容现有HTTP服务|
|`RAGFLOW_API_KEY`|远程检索必需的密钥，不记录在日志中|
|`RAGFLOW_TIMEOUT_SECONDS`|默认60秒，范围大于0且不超过120；单次连接/读取超时，不是整轮时限|
|`RAGFLOW_DATASET_NAME`|默认“售后规则知识库”，须唯一匹配|
|`RAGFLOW_ASSISTANT_NAME`|默认“售后助手”，须绑定上述知识库；只核对绑定，不另建助手会话|
|`AFTER_SALES_DB_PATH`|默认runtime/after_sales.sqlite3，保存模拟售后状态与去重事件|

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

### 6. 测试退货规则与幂等

配置RAGFlow后启动交互模式，使用当前用户所属订单。例如：

```powershell
uv run orderagent --user-id RETURN-HP-U-0004
```

输入“我要退货”，补充订单RETURN-HP-O-0004和“耳机左耳没有声音”等用户报告。RAG有支持模拟入口的规则时先展示订单/原因；单独回复“确认退货 RETURN-HP-O-0004”后创建一次模拟入口。无适用方案或知识服务异常立即转人工，不需要累计四次。

不再读取模拟商品完好、购买确认或签收七日等资格证据；问题依据用户描述和RAG判断。真实平台资格、质量鉴定、退货提交和退款仍未实现。

同订单跨会话复用有效入口，已退货/已退款拒绝再次办理，处理中或结果待核查阻止重复调用。创建前SQLite原子占用，创建异常不删除记录后盲目重试。普通咨询/人工记录不等于已经退货。

首次明确退货计为1，普通重复诉求第4次转人工；单纯补信息和确认不计次。质量仍一次方案后追问转人工，用户明确要求人工可立即转接。

## 架构与模块分工

```text
用户 → CLI → create_deep_agent / LangGraph ↔ ChatDeepSeek
                         ↓ 工具调用
                     ├─ query_order → 归属校验
                     │    └─ JsonOrderRepository → data/orders.json
                     ├─ query_products → 型号匹配／候选确认
                     │    └─ JsonProductRepository → data/products.json
                     ├─ query_quality_support → 一次方案或自动交接
                     │    └─ RagflowKnowledge → RAGFlow（未配置时使用本地JSON）
                     ├─ handle_return → 用户描述＋RAG、归属、确认、幂等 → 模拟入口／人工
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

退货接口位于[integrations/returns.py](src/orderagent/integrations/returns.py)，目前只提供create_link。旧模拟资格接口和ReturnEvidence已移除。入口方案依据用户描述和RAG，订单用于身份、商品定位与幂等。

核心校验函数为[check_return_operation](src/orderagent/tools/return_tools.py)，状态存储在[AfterSalesRepository](src/orderagent/repositories/after_sales_repository.py)的return_operations表。以用户＋订单唯一，创建前事务占用，稳定request_id不依赖会话或原因；已有有效入口复用，处理中/不明结果不重试，终态拒绝重复。URL仍检查HTTPS、允许域名、身份、模拟标记与有效期。

创建成功仅记录link_ready，不是退货完成。可信适配代码可写入returned/refunded/processing等阶段，目前没有真实平台同步或管理员重开接口；服务无法自行知道平台上新发生的退货状态。原子检查保证本地最多一个创建调用方，真实平台仍需服务端幂等，不能宣称已实现真实平台exactly-once。

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
│   ├── tools/return_tools.py       退货引导与入口工具工厂
│   ├── integrations/handoff.py     人工平台接口与模拟实现
│   ├── integrations/returns.py     退货资格与申请入口模拟接口
│   ├── repositories/order_repository.py  JSON仓储
│   ├── repositories/product_repository.py 商品只读匹配
│   ├── repositories/support_repository.py 故障知识匹配
│   ├── schemas/order_schema.py    参数、上下文和结果结构
│   ├── schemas/product_schema.py  商品参数及证据结构
│   ├── schemas/support_schema.py  质量参数、知识和会话阶段
│   ├── schemas/return_schema.py   退货参数、幂等操作记录和扩展会话状态
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
|[orders.json](data/orders.json)|7460|原60条、7200条批量测试订单与200条缺少完整资格证据的退货基础订单|
|[products.json](data/products.json)|379|商品查询运行数据：鼠标90、键盘89、耳机200|
|[product_faqs.json](data/product_faqs.json)|1140|后续商品事实问答资料|
|[support_knowledge.json](data/support_knowledge.json)|68|统一故障知识：原60条摘要与新增8条案例；其中3条为明确标注的项目模拟指导|
|[after_sales_cases.json](data/after_sales_cases.json)|2021|模拟售后开发案例|
|[tickets.json](data/tickets.json)|1518|原18条模拟工单与新增1500条关联测试工单；尚未接入运行工单服务|

商品、问答、故障知识和模拟案例合计3608条，模拟订单和工单另计。新增8条质量案例已并入故障知识，保留原记录ID、来源性质和适用条件。事实问答与商品主档共享证据，模拟案例共享场景模板，不应将总条数当作独立商品数或真实用户规模。

2026-10-23新增测试数据按每类3400条组织：2400条订单、500条工单、500条Markdown问答，三类共10200条。结构化数据追加到既有JSON；另外1500条问答保存在 [ragflow_knowledge](data/ragflow_knowledge)，包含180条商品事实问答、60条条件式故障问答和1260条售后规则场景。17份Markdown中有2份基础参考文档，不计入问答数量。

售后规则场景基于20个主题×21种条件构造，区分官方法规摘要与项目流程，不是1260条独立法规。法律适用范围、商品时效、模拟标记及来源均在正文中保留；没有制造新的商品参数、真实交易或品牌保修承诺。导入方式、数量口径和订单查询示例见 [数据使用说明](data/README.md)。知识文件已由用户导入RAGFlow，当前支持远程检索；问答不能同时作为独立评测集。

新增订单查询示例：

```powershell
uv run orderagent --user-id LOAD-HP-U-0001 --order-id LOAD-HP-O-000001
```

新增用户和订单按品类隔离，每类120个模拟用户。使用其他用户（包括 `DEMO-U-001`）查询上述订单应返回 `ORDER_NOT_FOUND`。订单金额及物流均为模拟，不能当作实际价格或平台执行结果。

另补200条基础退货订单，批次 `RETURN-BASIC-20261023-V1`，耳机67、鼠标67、键盘66条，覆盖未付款、待发货、运输中、近期签收及超期签收。这些订单仍作为基础查询/归属测试数据；当前不再要求它们提供模拟商品完好等资格证据，能否提供模拟入口取决于用户描述、RAG匹配、确认与幂等检查。

`data/` 另保留来源登记、演示规则、元数据、历史转换报告、来源许可和文件哈希清单。当前运行读取订单与商品；配置RAGFlow时检索远程故障/规则知识，未配置时读取本地统一故障知识集合；不把模拟售后案例或旧工单当作已验证的维修方案。新增案例来源自包含在各记录中。原 `support_cases.json` 已合并删除，知识文件路径统一通过 `SUPPORT_DATA_PATH` 配置，不再读取 `SUPPORT_CASES_PATH`。

订单字段中，`id` 为订单号，`user_id` 为所属用户，`items` 为商品明细，`shipping_status` 为发货状态，`logistics_status` 为物流状态，`snapshot_at` 为快照时间。

**数据边界**：订单和工单均为模拟记录；商品资料来自2016-05-09更新的历史开放目录，不代表当前在售、现价、库存或售后政策。原始英文参数未逐条翻译。

## 查询边界与会话行为

订单工具参数只允许订单号。用户身份从程序运行上下文注入，不允许模型通过额外参数切换用户；订单不存在与不属于当前用户统一返回 `ORDER_NOT_FOUND`。

商品工具只接受 `query`，精确ID、型号和源SKU优先，其后使用有限的中英文品类、品牌别名与关键词匹配。模型从问题中提取商品关键词；尚不提供模糊纠错、语义检索或推荐排序。详情依据原始 `description` 和 `evidence_groups`，保留来源与更新时间；空证据分组不能推断为不支持。当前售价、库存以及资料未记载的DPI、回报率、续航均不能补全。

`--user-id` 是本地演示上下文，不是真实身份认证。未来接入服务时，必须从可信登录会话获得用户身份，并保证不同用户使用不同线程。

现有 `BusinessToolPolicyMiddleware` 只开放 `query_order`、`query_products`、`query_quality_support`、`handle_return` 和 `handoff_to_human`，分别校验参数。涉及质量、退货或交接的并行工具批次会被拒绝，防止多份方案和阶段写入冲突。未启用本机文件系统后端；模型不能借助框架文件工具读取本地 `.env`。

故障匹配采用有限症状别名，不是语义搜索；有系统、品牌限制的方案需用户提供相应信息，未知时不推测。通用案例明确标注为项目模拟指导。自然语言反馈识别未做全面准确率评估，任意否定、引用或复杂多意图句仍可能误判；模型须遵循提示词补充识别未解决并交接。禁止第二次自动知识方案由程序状态控制。

检查点可能保留失败运行的部分状态，必要时用 `/reset` 清理。当前没有自写历史回滚或固定轮数截断；长会话总结效果尚未独立评测。

## 测试与检查

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

2026-10-26本次用户描述驱动退货与幂等更新后203项自动化测试通过，包含跨会话复用、两个Agent并发确认只创建一次、终态拒绝、结果不明不重试、用户隔离，以及既有质量一次方案、四次诉求、RAG异常和链接安全。旧模拟资格/签收期间测试已替换为新策略测试；Ruff检查及42个Python文件格式检查通过。

用户更新API后，2026-10-26复测RAGFlow已恢复：HTTP 200、业务码0、知识库/助手绑定正确、检索有结果。默认1200输出额度下退货规则选择连续出现长度截断；仅在诊断进程使用4096后调用完成，但样例仍未返回适用退货方案。质量完整流程一轮成功、一轮直接转人工，不能视为已稳定验证；业务代码未因本次测试修改。详见[连接复测日志](docs/2026-10-26+RAGFlow连接复测.md)。

当前锁定并安装的核心版本：

|依赖|版本|
|---|---|
|deepagents|0.7.29|
|langchain|1.4.6|
|langchain-core|1.6.12|
|langgraph|1.2.16|
|langchain-deepseek|1.1.1|
|ragflow-sdk|0.25.4|

## 开发约定与后续计划

在现有架构内优先最小修改。使用官方API前核对当前依赖版本；工具调用、执行、消息、重试和流式输出优先采用官方封装。确需自写模块时，先说明必要性与改动计划，等待确认后实现。完整要求见 [AGENTS.md](AGENTS.md)。

每次更新在 `docs/` 写日志，文件名为 `YYYY-MM-DD+内容.md`，包含更新内容、解决的问题、涉及文件、验证结果及限制，日期使用 Asia/Shanghai 时区。

后续功能候选为真实客服平台对接、生产数据库与多问题工单管理，具体实现另行讨论。当前没有网页、生产登录、真实电商接口、退货写入或人工通知。

## 官方参考

- [Deep Agents 自定义](https://docs.langchain.com/oss/python/deepagents/customization)
- [Deep Agents 后端](https://docs.langchain.com/oss/python/deepagents/backends)
- [ChatDeepSeek 集成](https://docs.langchain.com/oss/python/integrations/chat/deepseek)

## RAG规则与售后状态使用说明

配置RAGFlow地址后，服务异常、无适用知识或结构化结果不合法时，质量和退货均立即转人工，不等待第四次，不回退到模型常识或本地知识。未配置地址时保留原有本地知识演示模式。知识库与助手名称可配置，同名须唯一且绑定一致。

SDK固定为0.25.4。官方SDK没有超时参数，适配器仅补齐GET/POST超时；检索和列表仍使用官方接口，无自写网络重试。证据选择使用官方with_structured_output的json_mode，兼容DeepSeek思考模式；只展示可在检索分块中逐字找到的摘录，保留来源。适用性判断仍依赖模型，不能保证所有自然语言或知识污染均被正确识别。

每次find最多三个RAGFlow请求（知识库、助手、检索），有结果时额外调用一次证据选择模型。该调用发生在工具内部，不计入Agent模型节点的6次限制，但受工具调用次数间接约束，继承模型SDK的超时与重试；60秒不是整轮总耗时。

7460条订单全部初始化“未售后”，1518条工单仅为隔离的历史测试样本。质量咨询可不提供订单；需要更新订单时，须提供本人订单号并核验商品品类。未签收的到货故障交人工核实。SQLite记录“问题已反馈”“已提供一次排查”“已转人工（模拟）”等实际阶段，订单查询合并显示after_sales_status。

/reset只清空聊天，不删除SQLite记录。质量一次方案约束继续保留；退货不受一般咨询记录阻止，独立台账拒绝已退货/已退款，处理中阻止，有效入口跨会话复用；目前按订单记录，不能自动区分同一订单多个独立问题，也没有自动结案功能。SQLite记录不等于真实平台售后单，聊天历史仍使用内存检查点。

明确退货后核验订单归属，按用户报告匹配适用规则，不读模拟签收/资格证据。通过后先展示订单与原因，用户单独回复“确认退货 订单号”才生成测试链接。夹杂其他话语、引号或否定不能通过确认，改订单或原因需要重新确认。确认时重查规则和可信退货状态并原子占用；用户描述/RAG均不能写入退货终态或伪造质量鉴定。取消退货清除当前机器人引导，不代表撤销平台申请。

主要实现：[RAG适配器](src/orderagent/integrations/ragflow.py)、[售后状态仓储](src/orderagent/repositories/after_sales_repository.py)、[质量工具](src/orderagent/tools/support_tools.py)、[退货工具](src/orderagent/tools/return_tools.py)。运行目录runtime已忽略，不上传数据库与密钥。边界详见[当前规则](docs/rules.md)。

## 当前转人工策略（2026-10-26）

- 质量问题：提供一次排查后继续询问同一故障、反馈未解决或要求另一方案，立即转人工；无适用质量知识仍直接转人工。
- 退货问题：首次明确申请算第1次，同一订单累计第4次转人工；用户明确要求人工则立即转接。退货代办、坚持诉求、质量给过方案不会单独触发提前转接；RAG未命中适用方案或异常则立即转人工。
- is_return_request由模型区分本轮是否为明确退货请求；单纯补订单号/原因、确认、查询进度和引用/否定应传False。完整确认语由程序排除计数，同一HumanMessage.id不重复计数。
- 诉求次数保存在原官方会话状态中，按订单分别计数；退货操作单独保存在SQLite中；换理由、切换订单再切回、取消引导不会清零同订单计数。/reset或重启会清空线程计数，但不清除SQLite实际处理记录。
- RAG无适用知识或异常立即转人工；生成入口仍需归属、规则匹配、幂等检查和完整确认语，不再核对模拟资格证据。已生成的有效链接重复使用，过期引导后台，不重复创建。
- handoff_to_human增加topic与request_human参数，防止退货以普通交接工具绕过阈值；复杂意图和参数分类依赖模型，不能保证任意自然语言都识别正确。质量当前仍按线程阶段保守限制，尚无独立问题ID。

修改位置及验证见[本次策略日志](docs/2026-10-26+质量与退货转人工策略分离.md)，实际边界以[规则文档](docs/rules.md)为准。先前日志和已上传知识文档可能保留旧流程描述；程序使用本次策略，未自动修改远端知识库。

## 用户报告与退货幂等（2026-10-26）

处理依据改为用户描述＋适用RAG规则。去掉模拟资格证据、签收状态和固定七日硬门槛，保留身份、商品定位、确认语、链接安全与幂等。用户描述不是已经鉴定的事实；当前只生成模拟申请入口。

新增退货台账在原SQLite文件中自动建表，不清空旧数据。已提供入口的旧版记录若缺少可核验响应，会提示人工核查，不能凭空恢复或重复生成。creating/uncertain不会自动超时释放；人工核实、重新开启和真实平台状态同步需后续对接。

详见[本次更新日志](docs/2026-10-26+用户问题驱动退货与幂等校验.md)及[当前规则](docs/rules.md)。
