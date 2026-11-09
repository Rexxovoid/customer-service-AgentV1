# RAG导入与24题验收

版本：RAG-20261109-V3。本文是本地操作说明，不导入知识库。此次没有连接、上传、删除或解析远端文档。

## 导入范围与结构

仅上传`data/ragflow_knowledge/`当前17份Markdown：

| 文件 | 内容 |
|---|---|
| `00_售后规则依据与能力边界.md` | R01—R20法律/项目规则摘要，另有P-INFO-01陈述变化与P-CONTEXT-01条件族索引 |
| `00_品类售后材料清单.md` | 按任务收集必要信息与人工上下文，不强制全量索要型号/系统 |
| `HP/MS/KB_01_商品事实问答.md` | 180个历史商品单元，保留ID、参数证据、时效与来源 |
| `HP/MS/KB_02_故障条件问答.md` | 64个故障单元：症状别名→适用条件→必要信息→方案→限制→来源 |
| `HP/MS/KB_03—05_售后规则场景_*.md` | 每类20条规则，分R01—R07/R08—R14/R15—R20；文件名保留兼容原检索，正文不再重复420个场景 |

规则单元的品类副本复用同一法律依据，不等于新增法规。模型生成的演示指导与厂商摘要仍分别标明，不把模拟条目包装为厂商维修结论。

## 远端同步与解析检查

1. 先记录远端文档/知识版本及当前配置，再同步上述17份同名文件；避免旧版与新版一起检索，产生重复或旧边界。远端替换由使用者操作。
2. 不导入`data/evaluation/`、`data/archive/`、订单/工单JSON、manifest、验证报告或本操作说明。历史材料不是新增知识。
3. 按RAGFlow实际解析能力尝试按二级标题保留完整单元；不要假定Markdown/YAML会自动变成检索过滤元数据。先少量导入，再查看真实分块。
4. 同一分块应能核对条件、例外、步骤和来源；如果被拆散，先调整解析/分块配置，不放宽程序原文匹配。
5. 当前应用仅使用返回结果前5个非空片段、每片正文最多6000字符；退货候选还会排除明确命名的故障问答文档。检查实际送入选择模型的候选，不仅看远端相似度。
6. 助手必须绑定正确知识库，名称与配置一致。未同步或未解析完成时，不能把本地改动当作远端测试结果。

## 24题文件与运行

- 人工逐题标准：[RAG_ACCEPTANCE_24.md](evaluation/RAG_ACCEPTANCE_24.md)。
- 机器可读标准：[rag_acceptance_gold.json](evaluation/rag_acceptance_gold.json)，含文档/规则、适用条件、动作、必须项、禁止项及判定范围。
- 现有框架案例：[rag_acceptance_cases.jsonl](evaluation/rag_acceptance_cases.jsonl)，E43—E66，三类各8题、共30轮。全部pending、正文上传未授权。

```powershell
uv run orderagent-eval validate --cases data/evaluation/rag_acceptance_cases.jsonl
# 先确认RAGFLOW_API_URL、知识库/助手绑定及新版文档解析完成；以下会调用真实模型/RAG。
uv run orderagent-eval run --cases data/evaluation/rag_acceptance_cases.jsonl --ids E43,E44 --repeat 1
```

案例mode为live_agent，复用既有隔离会话与数据库；RAG URL留空会使用本地知识，不能把这种结果标成远程RAG验收。不要为查看正文把live_agent改为controlled_agent：两种模式的知识来源不同，当前Langfuse正文审批不开放live_agent。

如需看候选正文，临时开启`RAGFLOW_EVIDENCE_DEBUG=true`并重启入口，查看`evals/debug/rag_evidence/`的`candidates`、实际messages、structured_result与异常。已有20份上限，达到上限需使用者自行管理文件；不要擅删历史实验。调试正文可能包含用户和知识内容，审查后关闭。

## 评分和人工核查

- E43/E44、E51/E52、E59/E60为三组独立改写的同义对照，比较同样条件下的规则与动作，不要求逐字相同答案。
- 不适用题所列规则是**应排除的候选**，不是必须命中的正例。若存在其他独立且适用来源，允许有依据回答或补问；不能仅凭预期文件未命中判业务失败。
- E48/E56/E64没有型号技术依据，技术答案为unjudgeable；不得给料号、温度、固件或口令造金标。不编造/说明缺据/人工核实可以另外审查。
- 五维裁判的reference写明无法判定时技术分应null；这是模型约束，不是新增的程序强制。人工仍需检查，不能把模型返回数值当成有了原始证据。
- 自动checks仅覆盖声明的工具路由、规则咨询非申请及creates=0；不证明内容、补问相关性、规则适用、历史与版本更新全部正确。其余看gold及执行证据，填写既有human_review.csv后compare。
- 多轮申请题不保证真实RAG一定判可退；用户陈述未核验，缺必要条件补问、规则不支持交接需有依据。没有有效执行确认，所有题都不能创建入口。
- 本轮题目是本次文档整理后的开发验收集，未入库，但不能据此宣称严格盲测或独立最终测试集；最终评测应另留题目与规则族。

## 前后对照

若只测检索，使用派生集`data/evaluation/rag_retrieval_cases.jsonl`（E67—E90，均pending），运行时不加--judge。它绕过主Agent信息提取与规则选择，仅复用原检索和过滤；多轮题只取首轮请求，不能代替多轮验收。可分别查看binding_metrics与retrieval_metrics；命中只在人工审批的正例文档上统计，详情见[测评指南](../evals/EVALUATION_GUIDE.md#纯rag检索性能)。

固定模型/规则选择配置、问题、初态与时钟口径，记录代码/案例/知识版本、实际候选、选择结果、错误和阶段耗时。对比正确知识命中、条件完整、正确动作、不必要补问、错误放行/误人工及P50/P95。没有实跑前不填写改善百分比；历史实验不追改，新旧文档对照分开建实验。

原17份文档与1260个旧场景ID映射在`data/archive/ragflow_knowledge/2026-11-09-before-v3/`，仅供审计，不导入。JSON业务知识未改，本地JSON与远端Markdown是不同来源，验收时必须注明。
