# 业务数据、RAG材料与独立测评

更新：2026-11-09。当前程序边界见[规则总表](../docs/rules.md)；远程知识同步与验收见[RAG导入指南](RAG_IMPORT_GUIDE.md)。

## 当前数据分工

| 文件/目录 | 用途与限制 |
|---|---|
| products.json（379）/ product_faqs.json（1140） | 有来源的历史商品资料及衍生问答，不代表实时售价/库存/保修 |
| support_knowledge.json（72） | 本地故障知识；来源摘要与项目模拟指导分开，不与远端内容自动同步 |
| after_sales_cases.json（2021） | 模拟案例，不是实时售后状态 |
| orders.json（7460） | 模拟订单/物流快照，初始未售后；精确查询需订单号及正确演示身份 |
| tickets.json（1518） | 历史模拟工单，不是在线客服队列，不驱动订单状态 |
| ragflow_knowledge/ | 当前17份Markdown导入材料，包含180个商品、64个故障及60个品类规则单元，基础文档另计 |
| evaluation/ | 独立测评、参考标准与清单；不得导入知识库 |
| archive/ragflow_knowledge/2026-11-09-before-v3/ | 原17份RAG文档、旧数据说明及ID映射，仅供历史审计，不导入 |
| metadata.json / manifest.json | 当前数量口径及字节/SHA256清单；evaluation另有独立清单 |

旧扩充批次为每类2400订单、500工单和500问答，三类10200条混合数据，不是独立事实规模。旧1500条问答后增加4条故障知识，整理前为1504条。此次1260个条件模板合并为60个品类规则单元，当前带RAG ID共304条；历史批次统计在metadata中保留，不再当作当前导入数量。

## 订单与售后数据

```powershell
uv run orderagent --user-id LOAD-HP-U-0001 --order-id LOAD-HP-O-000001
uv run orderagent --user-id LOAD-MS-U-0002 --order-id LOAD-MS-O-000002
uv run orderagent --user-id LOAD-KB-U-0003 --order-id LOAD-KB-O-000003
```

HP/MS/KB分别为耳机/鼠标/键盘。订单归属由工具核对，错误用户不能读取。金额、物流号、用户与工单均为模拟，不含真实个人资料。

其中200条RETURN-BASIC订单只提供基础交易/物流字段，不包含完整资格证据；当前程序不读取这些模拟资格字段决定质量或退货权利，而是结合用户陈述、适用RAG规则及程序身份/确认/状态校验。MOCK_RETURN_SCENARIO仅模拟入口响应，不能补写真实审核事实；当前不将固定JSON日期作为统一七日门槛。

质量咨询不登记为订单售后。实际退货引导/幂等状态在SQLite，当前只提供模拟入口，不代表提交、批准、完成或退款。结束/重置咨询不删除可信业务台账。法定权利与项目模拟能力不能混为一谈。

## 知识优化与来源

- 故障文档按症状/别名、适用条件、必要信息、方案、限制及来源组织，不新增未经核对的维修步骤。
- 规则文档保留R01—R20及来源，以品类规则替代条件组合堆叠；新ID如RAG-MS-R02，旧场景ID到规则的映射保留在历史manifest。
- 同名文件沿用以兼容现有检索命名过滤，正文版本为RAG-20261109-V3。文档仅是知识，不替代程序权限与确认边界。
- 本次重点核对Apple单耳无声、Microsoft USB输入设备、Logitech跨程序检查及市场监管总局规则来源；其他条目保留原来源与核对记录，未声称重新逐项认证。
- 未上传或修改远端RAGFlow。只同步ragflow_knowledge当前文件，先验证分块/解析/绑定，不让旧版与新版同时命中。

## 独立24题验收

新增[E43—E66人工验收标准](evaluation/RAG_ACCEPTANCE_24.md)、[结构化标准](evaluation/rag_acceptance_gold.json)和[可运行案例](evaluation/rag_acceptance_cases.jsonl)，每类8题、共30轮。包含典型故障、同义、缺项、完整条件、不适用、无答案、退货例外及多轮修改；3题技术答案无法判定。

全部pending、未授权正文上传；旧40例及审批不改，总计64例。机器checks不覆盖全部自然语言标准，未检查项不能视为通过。题目不入库，本轮是开发验收而非严格盲测，最终实验仍需独立留题。

另有24条E67—E90纯检索派生请求，文件为evaluation/rag_retrieval_cases.jsonl，仍全部pending；仅取前述24题首轮，不测主模型提取/规则选择/多轮状态，不加--judge。含派生集共88条运行案例，不等于88道独立问题；详见测评指南的纯检索性能章节。

操作、候选查看及评分限制见[RAG导入指南](RAG_IMPORT_GUIDE.md)和[测评指南](../evals/EVALUATION_GUIDE.md)。本次只做本地整理与静态校验，没有真实检索或效果提升结论。
