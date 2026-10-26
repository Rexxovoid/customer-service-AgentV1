import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from orderagent.schemas.return_schema import ReturnOperation


class AfterSalesRepository:
    """保存模拟售后当前状态与事件；订单快照仍保持只读，不记录聊天全文。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS after_sales ("
                "user_id TEXT, order_id TEXT, status TEXT NOT NULL, updated_at TEXT, "
                "PRIMARY KEY(user_id, order_id))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS after_sales_events ("
                "event_id TEXT PRIMARY KEY, user_id TEXT, order_id TEXT, "
                "status TEXT, created_at TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS return_operations ("
                "user_id TEXT NOT NULL, order_id TEXT NOT NULL, request_id TEXT NOT NULL, "
                "status TEXT NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY(user_id, order_id))"
            )

    def _connect(self):
        """每次操作独立连接，使用SQLite事务和五秒锁等待。"""
        return sqlite3.connect(self.path, timeout=5)

    def get_status(self, user_id: str, order_id: str) -> str:
        """无运行记录的订单统一返回未售后，绝不从历史工单推断。"""
        with self._connect() as connection:
            operation = connection.execute(
                "SELECT payload FROM return_operations WHERE user_id=? AND order_id=?",
                (user_id, order_id),
            ).fetchone()
            row = connection.execute(
                "SELECT status FROM after_sales WHERE user_id=? AND order_id=?",
                (user_id, order_id),
            ).fetchone()
        if row and row[0].casefold() in {
            "returned",
            "refunded",
            "已退货",
            "已退款",
            "退货完成",
            "退款完成",
            "returning",
            "return_processing",
            "refund_processing",
            "退货中",
            "退款中",
            "退货处理中",
            "退款处理中",
            "退货申请处理中",
        }:
            return row[0]
        if operation:
            record = ReturnOperation.model_validate_json(operation[0])
            return {
                "creating": "退货入口生成中",
                "link_ready": "已提供退货测试入口" if record.simulated else "已提供退货入口",
                "processing": "退货处理中",
                "uncertain": "退货操作待核查",
                "returned": "已退货",
                "refunded": "已退款",
            }[record.status]
        return row[0] if row else "未售后"

    def record(self, user_id: str, order_id: str, status: str, event_id: str) -> None:
        """以工具调用/消息ID去重，原子写入事件和当前状态；仅由已核验归属的调用方使用。"""
        timestamp = datetime.now(timezone.utc).isoformat()
        event_id = json.dumps([user_id, order_id, event_id])
        with self._connect() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO after_sales_events VALUES (?, ?, ?, ?, ?)",
                (event_id, user_id, order_id, status, timestamp),
            ).rowcount
            if inserted:
                connection.execute(
                    "INSERT INTO after_sales VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(user_id, order_id) DO UPDATE SET "
                    "status=excluded.status, updated_at=excluded.updated_at "
                    "WHERE lower(after_sales.status) NOT IN "
                    "('returned','refunded','已退货','已退款','退货完成','退款完成') AND "
                    "(lower(after_sales.status) NOT IN "
                    "('returning','return_processing','refund_processing','退货中','退款中',"
                    "'退货处理中','退款处理中','退货申请处理中') OR lower(excluded.status) IN "
                    "('returned','refunded','已退货','已退款','退货完成','退款完成',"
                    "'returning','return_processing','refund_processing','退货中','退款中',"
                    "'退货处理中','退款处理中','退货申请处理中')) AND "
                    "(excluded.status NOT IN ('问题已反馈', '已提供一次排查') "
                    "OR after_sales.status IN ('问题已反馈', '已提供一次排查'))",
                    (user_id, order_id, status, timestamp),
                )

    def start_issue(self, user_id: str, order_id: str, event_id: str) -> bool:
        """原子占用首次排查，避免不同线程同时为同一订单给出多次方案。"""
        timestamp = datetime.now(timezone.utc).isoformat()
        event_id = json.dumps([user_id, order_id, event_id])
        with self._connect() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO after_sales VALUES (?, ?, ?, ?)",
                (user_id, order_id, "问题已反馈", timestamp),
            ).rowcount
            if inserted:
                connection.execute(
                    "INSERT INTO after_sales_events VALUES (?, ?, ?, ?, ?)",
                    (event_id, user_id, order_id, "问题已反馈", timestamp),
                )
        return bool(inserted)

    def get_return_operation(self, user_id: str, order_id: str) -> ReturnOperation | None:
        """仅按用户和订单读取操作台账，缺少记录保持未知，不从普通咨询推断已退货。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM return_operations WHERE user_id=? AND order_id=?",
                (user_id, order_id),
            ).fetchone()
        return ReturnOperation.model_validate_json(row[0]) if row else None

    def claim_return_operation(self, operation: ReturnOperation) -> bool:
        """事务内检查旧终态并原子占用订单；并发和跨会话只有一个调用方能创建入口。"""
        operation = ReturnOperation.model_validate(operation.model_dump())
        if operation.status != "creating" or operation.response is not None:
            raise ValueError("只能占用尚未执行的创建操作")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM after_sales WHERE user_id=? AND order_id=?",
                (operation.user_id, operation.order_id),
            ).fetchone()
            if row and row[0].casefold() in {
                "returned",
                "refunded",
                "已退货",
                "已退款",
                "退货完成",
                "退款完成",
                "returning",
                "return_processing",
                "refund_processing",
                "退货处理中",
                "退货中",
                "退款中",
                "退款处理中",
                "退货申请处理中",
                "已提供退货测试入口",
                "已提供退货入口",
                "退货操作待核查",
            }:
                return False
            return bool(
                connection.execute(
                    "INSERT OR IGNORE INTO return_operations VALUES (?, ?, ?, ?, ?)",
                    (
                        operation.user_id,
                        operation.order_id,
                        operation.request_id,
                        operation.status,
                        operation.model_dump_json(),
                    ),
                ).rowcount
            )

    def update_return_operation(self, operation: ReturnOperation) -> None:
        """由可信业务路径写入操作结果；请求必须对应，完成/退款终态不可回退。"""
        operation = ReturnOperation.model_validate(operation.model_dump())
        if operation.status == "creating":
            raise ValueError("不能通过结果更新重新占用")
        response = operation.response
        if operation.status == "link_ready" and (
            response is None
            or response.status != "ready"
            or response.user_id != operation.user_id
            or response.order_id != operation.order_id
            or response.request_id != operation.request_id
            or response.simulated != operation.simulated
        ):
            raise ValueError("入口响应与操作记录不一致")
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE return_operations SET status=?, payload=? "
                "WHERE user_id=? AND order_id=? AND request_id=? "
                "AND status NOT IN ('returned','refunded')",
                (
                    operation.status,
                    operation.model_dump_json(),
                    operation.user_id,
                    operation.order_id,
                    operation.request_id,
                ),
            ).rowcount
            if not changed:
                raise ValueError("操作不存在、请求不一致或已处于终态")
