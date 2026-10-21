import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from orderagent.schemas.order_schema import StoredOrder


class DataLoadError(ValueError):
    pass


class OrderRepository(Protocol):
    def find_owned(self, order_id: str, user_id: str) -> StoredOrder | None: ...


class JsonOrderRepository:
    """Read-only adapter. Can later be replaced with a database/platform adapter."""

    def __init__(self, path: Path):
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
        order = self._orders.get(order_id)
        if order is None or order.user_id != user_id:
            return None
        return order.model_copy(deep=True)
