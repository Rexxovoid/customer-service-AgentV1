from langchain.tools import tool
from langchain_core.tools import BaseTool

from orderagent.repositories.product_repository import JsonProductRepository
from orderagent.schemas.order_schema import ToolResult
from orderagent.schemas.product_schema import ProductQuery


class ProductQueryTool:
    """组织商品查询结果，并提供官方LangChain工具入口。"""

    name = "query_products"
    description = (
        "当用户询问耳机、鼠标、键盘的型号、产品介绍或具体性能参数时调用，无需订单号。"
        "query仅包含用户问题中的品类、品牌、型号、商品名或商品ID关键词，"
        "例如鼠标、罗技 M185、MS-P-0002，不把整段问题当作检索词。"
        "唯一匹配返回详情；多条匹配最多返回5款候选及总匹配数，"
        "needs_clarification为true时先分别展示已有参数并请用户确认型号，禁止混合参数。"
        "根据description和evidence_groups解释性能，未记载的DPI、回报率或续航必须说明未知；"
        "空证据分组不表示不支持。资料为2016年历史目录，当前售价和库存未知。"
        "查已购订单用query_order；实际故障排查用query_quality_support，"
        "用户反馈已给方案未解决时用handoff_to_human，不用本工具继续找维修方案。"
    )

    def __init__(self, repository: JsonProductRepository):
        """注入只读商品仓储，避免工具内部重复创建数据访问对象。"""
        self.repository = repository

    def execute(self, arguments: ProductQuery) -> ToolResult:
        """匹配商品并组织详情或最多5款候选，同时附带资料时效和参数边界。"""
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
        """用官方装饰器声明参数结构和工具描述，生成可注册的商品查询工具。"""

        @tool(
            self.name,
            description=self.description,
            args_schema=ProductQuery,
            response_format="content_and_artifact",
        )
        def query_products(query: str) -> tuple[str, dict]:
            """将查询结果同时转换为模型可读JSON和结构化artifact。"""
            result = self.execute(ProductQuery(query=query))
            return result.model_dump_json(), result.model_dump(mode="json")

        return query_products
