"""用户可见聊天与模拟人工路由；不保存或恢复Agent检查点。"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class ChatConflict(ValueError):
    """会话只读、忙碌或消息标识冲突，禁止自动重新执行。"""


class ChatRepository:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, owner TEXT NOT NULL,
                    title TEXT NOT NULL, channel TEXT NOT NULL DEFAULT 'agent',
                    status TEXT NOT NULL DEFAULT 'agent_consulting',
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    closed_at REAL, closed_by TEXT, close_reason TEXT,
                    version INTEGER NOT NULL DEFAULT 0, busy TEXT,
                    handoff_id TEXT, agent_id TEXT, handoff_status TEXT, human_started_at REAL,
                    last_user_at REAL, last_human_at REAL,
                    waiting_user INTEGER NOT NULL DEFAULT 0, waiting_since REAL);
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    conversation_id TEXT NOT NULL, client_id TEXT NOT NULL,
                    role TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL, request_id TEXT,
                    UNIQUE(conversation_id, client_id, role));
                CREATE INDEX IF NOT EXISTS chat_user_updated ON conversations(user_id,updated_at);
                CREATE INDEX IF NOT EXISTS chat_messages ON messages(conversation_id,seq);
            """)
            # 单实例启动恢复：不把未完成调用判成功或自动重放。
            db.execute("UPDATE messages SET status='unknown' WHERE status='processing'")
            db.execute("UPDATE conversations SET busy=NULL WHERE busy IS NOT NULL")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _owned(self, db, user, conversation):
        row = db.execute(
            "SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation, user)
        ).fetchone()
        if row is None:
            raise LookupError("会话不存在或无权访问")
        return dict(row)

    def create(self, user, owner):
        key, now = uuid4().hex, time.time()
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations(id,user_id,owner,title,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (key, user, owner, "新咨询", now, now),
            )
        return key

    def get(self, user, conversation):
        with self.connect() as db:
            return self._owned(db, user, conversation)

    def list(self, user, offset=0, limit=20):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC,id "
                    "LIMIT ? OFFSET ?",
                    (user, limit, offset),
                )
            ]

    def messages(self, user, conversation, before=None, limit=100):
        with self.connect() as db:
            self._owned(db, user, conversation)
            rows = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? AND seq<? "
                "ORDER BY seq DESC LIMIT ?",
                (conversation, before or 9223372036854775807, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def _message(self, db, conversation, client, role, text, status, request_id):
        db.execute(
            "INSERT INTO messages(id,conversation_id,client_id,role,text,status,"
            "created_at,request_id) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (uuid4().hex, conversation, client, role, text, status, time.time(), request_id),
        )

    def reserve(self, user, conversation, owner, client, text, request_id):
        """先保存用户消息并占用会话；重复ID只复用结果，不重试不明调用。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._owned(db, user, conversation)
            old = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? AND client_id=? AND role='user'",
                (conversation, client),
            ).fetchone()
            if old:
                if old["text"] != text:
                    raise ChatConflict("消息标识已用于其他内容")
                answer = db.execute(
                    "SELECT * FROM messages WHERE conversation_id=? AND client_id=? "
                    "AND role!='user'",
                    (conversation, client),
                ).fetchone()
                if old["status"] != "complete" or not answer:
                    raise ChatConflict("该消息处理中、失败或结果待核查，不自动重试")
                return row, {
                    "answer": answer["text"],
                    "role": answer["role"],
                    "request_id": answer["request_id"],
                    "replayed": True,
                }
            if row["status"] == "closed":
                raise ChatConflict("会话已结束，仅供查看，请开启新咨询")
            if row["channel"] == "agent" and row["owner"] != owner:
                raise ChatConflict("旧Agent会话仅供查看，请开启新咨询")
            if row["status"] == "transfer_failed":
                raise ChatConflict("交接失败，不自动重复交接；请开启新咨询")
            if row["busy"]:
                raise ChatConflict("该会话正在处理消息，请稍后再试")
            self._message(db, conversation, client, "user", text, "processing", request_id)
            db.execute(
                "UPDATE conversations SET busy=?,version=version+1,updated_at=?,last_user_at=?, "
                "waiting_user=0,waiting_since=NULL,"
                "title=CASE WHEN title='新咨询' THEN ? ELSE title END "
                "WHERE id=?",
                (client, time.time(), time.time(), text[:40], conversation),
            )
            return row, None

    def complete(
        self,
        user,
        conversation,
        client,
        text,
        role,
        request_id,
        handoff=None,
        closed=False,
        waiting_user=False,
    ):
        """回复与渠道变化同事务保存；仅结构化模拟交接结果可进入人工。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._owned(db, user, conversation)
            if row["busy"] != client or row["status"] == "closed":
                raise ChatConflict("会话状态变化，结果待核查")
            self._message(db, conversation, client, role, text, "complete", request_id)
            db.execute(
                "UPDATE messages SET status='complete' WHERE conversation_id=? "
                "AND client_id=? AND role='user'",
                (conversation, client),
            )
            now = time.time()
            db.execute(
                "UPDATE conversations SET busy=NULL,updated_at=?,version=version+1 WHERE id=?",
                (now, conversation),
            )
            if handoff:
                connected = handoff.status == "connected" and handoff.simulated and handoff.agent_id
                db.execute(
                    "UPDATE conversations SET channel=?,status=?,handoff_id=?,agent_id=?,"
                    "handoff_status=?,human_started_at=? WHERE id=?",
                    (
                        "human" if connected else "agent",
                        "human_active" if connected else "transfer_failed",
                        handoff.request_id,
                        handoff.agent_id,
                        handoff.status,
                        now if connected else None,
                        conversation,
                    ),
                )
            if role == "simulated_agent":
                db.execute(
                    "UPDATE conversations SET last_human_at=?,waiting_user=?,waiting_since=? "
                    "WHERE id=?",
                    (now, int(waiting_user), now if waiting_user else None, conversation),
                )
            if closed:
                self._close(db, conversation, "simulated_agent", "模拟人工主动结束", request_id)

    def uncertain(self, user, conversation, client):
        with self.connect() as db:
            self._owned(db, user, conversation)
            db.execute(
                "UPDATE messages SET status='unknown' WHERE conversation_id=? AND client_id=? "
                "AND role='user' AND status='processing'",
                (conversation, client),
            )
            db.execute(
                "UPDATE conversations SET busy=NULL WHERE id=? AND busy=?", (conversation, client)
            )

    def _close(self, db, conversation, by, reason, request_id):
        now = time.time()
        db.execute(
            "UPDATE conversations SET status='closed',closed_at=?,closed_by=?,close_reason=?,"
            "waiting_user=0,waiting_since=NULL,version=version+1,updated_at=? WHERE id=?",
            (now, by, reason, now, conversation),
        )
        self._message(
            db,
            conversation,
            "close",
            "system",
            "本次模拟人工服务已结束，历史记录仍可查看。",
            "complete",
            request_id,
        )

    def close(self, user, conversation, request_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._owned(db, user, conversation)
            if row["status"] == "closed":
                return row
            if row["busy"]:
                raise ChatConflict("消息正在处理中，暂不能结束服务")
            if row["channel"] != "human":
                raise ChatConflict("当前不是人工会话")
            self._close(db, conversation, "user", "用户主动结束", request_id)
            return self._owned(db, user, conversation)
