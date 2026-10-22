from langchain.tools import tool
from langchain_core.tools import BaseTool

from orderagent.repositories.product_repository import JsonProductRepository
from orderagent.schemas.order_schema import ToolResult
from orderagent.schemas.product_schema import ProductQuery


class ProductQueryTool:
    name = "query_products"
    description = (
        "查询耳机、鼠标、键盘的具体参数和性能。query仅提取商品实体关键词，"
        "如鼠标、罗技 M185或商品ID；多个结果是候选，须确认型号，不可混合参数。"
        "结果包含原始描述和参数证据；资料未提供的数值必须说明未知。"
    )

    def __init__(self, repository: JsonProductRepository):
        self.repository = repository

    def execute(self, arguments: ProductQuery) -> ToolResult:
        matches = self.repository.search(arguments.query)
        if not matches:
            return ToolResult.failure(
                "PRODUCT_NOT_FOUND", "未找到匹配商品，请提供品类、品牌或准确型号。"
            )
        return ToolResult(
            ok=True,
            data={
                "query": arguments.query,
                "match_type": "detail" if len(matches) == 1 else "candidates",
                "needs_clarification": len(matches) > 1,
                "total_matches": len(matches),
                "returned_count": min(len(matches), 5),
                "products": [row.model_dump(mode="json") for row in matches[:5]],
                "data_source": "local_historical_product_json",
                "notice": "商品来自2016年历史目录；当前售价和库存未知，不代表当前在售资料。",
                "parameter_scope": (
                    "参数证据分组不完整，空分组不代表不支持；完整description也须核对。"
                    "不能补全未记载的DPI、回报率、续航等数值，不能跨型号混合。"
                ),
            },
        )

    def as_langchain_tool(self) -> BaseTool:
        @tool(
            self.name,
            description=self.description,
            args_schema=ProductQuery,
            response_format="content_and_artifact",
        )
        def query_products(query: str) -> tuple[str, dict]:
            result = self.execute(ProductQuery(query=query))
            return result.model_dump_json(), result.model_dump(mode="json")

        return query_products
