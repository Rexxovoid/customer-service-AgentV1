import json
from typing import Literal

import requests
from pydantic import BaseModel, Field
from ragflow_sdk import RAGFlow

from orderagent.prompts.order_prompt import load_prompt
from orderagent.schemas.support_schema import SupportKnowledge


class TimeoutRAGFlow(RAGFlow):
    """仅补齐0.25.4官方SDK未暴露的请求超时，不重写检索或会话协议。"""

    def __init__(self, api_key: str, base_url: str, timeout: float):
        super().__init__(api_key=api_key, base_url=base_url)
        self.timeout = timeout

    def get(self, path, params=None, json=None):
        """保留官方GET协议，增加连接/读取超时，不配置重试。"""
        return requests.get(
            self.api_url + path,
            params=params,
            json=json,
            headers=self.authorization_header,
            timeout=self.timeout,
        )

    def post(self, path, json=None, stream=False, files=None):
        """保留官方POST协议，增加连接/读取超时，不配置重试。"""
        return requests.post(
            self.api_url + path,
            json=json,
            stream=stream,
            files=files,
            headers=self.authorization_header,
            timeout=self.timeout,
        )


class RuleSelection(BaseModel):
    """模型只能选择适用原文；不允许生成脱离来源的维修步骤或资格证据。"""

    applicable: bool = Field(description="是否有直接适用且足够完整的安全处理知识")
    chunk_id: str = Field(description="所选分块ID；不适用时为空")
    excerpt: str = Field(description="逐字摘录该分块中完整且带条件的答案；不改写")
    handling: Literal["basic_troubleshooting", "return_eligible", "human"] = Field(
        description="按用户描述匹配的处理方式；return_eligible只表示支持模拟入口，不是已核验资格"
    )


class RagflowKnowledge:
    """通过官方检索获取证据，再用官方结构化输出选择原文，不创建第二个执行Agent。"""

    def __init__(self, settings, model):
        self.client = TimeoutRAGFlow(
            settings.ragflow_api_key.get_secret_value(),
            settings.ragflow_api_url,
            settings.ragflow_timeout_seconds,
        )
        self.dataset_name = settings.ragflow_dataset_name
        self.assistant_name = settings.ragflow_assistant_name
        self.selector = model.with_structured_output(RuleSelection, method="json_mode")

    def find(self, query):
        """仅检索指定库并核对助手绑定；异常由业务工具转人工，禁止回退常识答案。"""
        datasets = self.client.list_datasets(name=self.dataset_name)
        datasets = [item for item in datasets if item.name == self.dataset_name]
        chats = self.client.list_chats(name=self.assistant_name)
        chats = [item for item in chats if item.name == self.assistant_name]
        if len(datasets) != 1 or len(chats) != 1 or datasets[0].id not in chats[0].dataset_ids:
            raise ValueError("知识库或助手绑定不唯一/不匹配")
        chunks = self.client.retrieve(
            dataset_ids=[datasets[0].id],
            question=(
                f"{dict(headphones='耳机', mice='鼠标', keyboards='键盘')[query.category]} "
                f"{query.issue} {query.environment}"
            ),
            page_size=5,
            top_k=64,
            similarity_threshold=0.3,
        )
        evidence = {
            str(item.id): {"content": item.content[:6000], "document": item.document_name}
            for item in chunks[:5]
            if item.content
        }
        if not evidence:
            return None
        selection = self.selector.invoke(
            [
                ("system", load_prompt("rag_selection_prompt")),
                (
                    "human",
                    json.dumps(
                        {"query": query.model_dump(), "evidence": evidence}, ensure_ascii=False
                    ),
                ),
            ]
        )
        selection = RuleSelection.model_validate(selection)
        chosen = evidence.get(selection.chunk_id)
        if (
            not selection.applicable
            or chosen is None
            or selection.handling == "human"
            or not selection.excerpt.strip()
            or selection.excerpt not in chosen["content"]
            or len(selection.excerpt) > 2500
        ):
            return None
        return SupportKnowledge(
            id=selection.chunk_id,
            category=query.category,
            question=query.issue,
            answer=selection.excerpt,
            applicability="仅适用所引原文明确列出的条件；未知条件需核实",
            source={
                "reference_text": f"RAGFlow / {chosen['document']} / {selection.chunk_id}",
                "handling": selection.handling,
            },
            data_origin="ragflow_retrieval",
        )
