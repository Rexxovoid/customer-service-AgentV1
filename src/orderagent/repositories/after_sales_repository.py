import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


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

    def _connect(self):
        """每次操作独立连接，使用SQLite事务和五秒锁等待。"""
        return sqlite3.connect(self.path, timeout=5)

    def get_status(self, user_id: str, order_id: str) -> str:
        """无运行记录的订单统一返回未售后，绝不从历史工单推断。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM after_sales WHERE user_id=? AND order_id=?",
                (user_id, order_id),
            ).fetchone()
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
                    "WHERE excluded.status NOT IN ('问题已反馈', '已提供一次排查') "
                    "OR after_sales.status IN ('问题已反馈', '已提供一次排查')",
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
