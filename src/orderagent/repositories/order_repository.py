import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from orderagent.schemas.order_schema import StoredOrder


class DataLoadError(ValueError):
    """数据文件无法读取或结构不合法时使用的可展示异常。"""

    pass


class OrderRepository(Protocol):
    """订单数据访问契约，便于以后替换数据库或电商平台适配器。"""

    def find_owned(self, order_id: str, user_id: str) -> StoredOrder | None:
        """返回指定用户所属的订单，不存在或无权访问时返回None。"""
        ...


class JsonOrderRepository:
    """从本地JSON加载的只读订单仓储，使用订单号字典索引。"""

    def __init__(self, path: Path):
        """加载并校验全部订单，建立索引并拒绝重复订单号。"""
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, list):
                raise ValueError()
            records = [StoredOrder.model_validate(row) for row in raw]
            self._orders = {row.id: row for row in records}
            if len(self._orders) != len(records):
                raise ValueError()
        except (OSError, ValueError, ValidationError):
            raise DataLoadError("订单数据无法加载，请检查文件格式、必需字段与重复ID。") from None

    def find_owned(self, order_id: str, user_id: str) -> StoredOrder | None:
        """按订单号精确取值并校验归属，返回深拷贝以保护仓储内部记录。"""
        order = self._orders.get(order_id)
        if order is None or order.user_id != user_id:
            return None
        return order.model_copy(deep=True)
