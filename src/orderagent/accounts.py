"""本地模拟账号初始化与SQLite映射；密码算法交给pwdlib，不参与Agent执行。"""

import argparse
import json
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from pwdlib import PasswordHash

from orderagent.settings import PROJECT_ROOT

AUTH_DIR = PROJECT_ROOT / "runtime" / "auth"
DEFAULT_DB = AUTH_DIR / "accounts.sqlite3"
DEFAULT_CREDENTIALS = AUTH_DIR / "demo_credentials.json"


class AccountStore:
    """独立账号库；每次操作独立连接，唯一约束保护用户与账号映射。"""

    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.passwords = PasswordHash.recommended()
        self.dummy_hash = self.passwords.hash(secrets.token_urlsafe(24))
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS accounts (
                username TEXT PRIMARY KEY, user_id TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
                version INTEGER NOT NULL DEFAULT 1, synthetic INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL, updated_at REAL NOT NULL)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, username):
        with self.connect() as db:
            row = db.execute("SELECT * FROM accounts WHERE username=?", (username,)).fetchone()
            return dict(row) if row else None

    def authenticate(self, username, password):
        row = self.get(username)
        try:
            valid = self.passwords.verify(
                password, row["password_hash"] if row else self.dummy_hash
            )
        except Exception:
            valid = False
        return row if row and row["enabled"] and valid else None

    def reset(self, username, password):
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE accounts SET password_hash=?, version=version+1, updated_at=? "
                "WHERE username=? AND synthetic=1",
                (self.passwords.hash(password), time.time(), username),
            )
            if cursor.rowcount != 1:
                raise ValueError("模拟账号不存在")


def inspect_users(data_dir):
    """仅订单用户建号；工单必须能关联同一订单用户，场景知识没有用户不算坏记录。"""
    rows = json.loads((data_dir / "orders.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("订单数据必须为数组")
    users, orders, issues = {}, {}, []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            issues.append({"kind": "invalid_order_record", "index": index})
            continue
        if row["id"] in orders:
            issues.append({"kind": "duplicate_order_id", "index": index})
            continue
        user = row.get("user_id")
        if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", user):
            issues.append({"kind": "invalid_order_user", "index": index})
            continue
        if row.get("synthetic") is not True:
            issues.append({"kind": "non_simulated_order", "index": index})
            continue
        entry = users.setdefault(user, {"order_count": 0, "example_order_ids": []})
        entry["order_count"] += 1
        if len(entry["example_order_ids"]) < 3:
            entry["example_order_ids"].append(row["id"])
        orders[row["id"]] = user
    for filename in ("tickets.json", "after_sales_cases.json"):
        path = data_dir / filename
        if not path.exists():
            continue
        referenced = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(referenced, list):
            raise ValueError("关联数据必须为数组")
        for index, row in enumerate(referenced):
            if not isinstance(row, dict):
                issues.append(
                    {"kind": "invalid_reference_record", "file": filename, "index": index}
                )
                continue
            if "user_id" in row and row["user_id"] not in users:
                issues.append({"kind": "orphan_user", "file": filename, "index": index})
            if "order_id" in row and orders.get(row["order_id"]) != row.get("user_id"):
                issues.append({"kind": "order_user_mismatch", "file": filename, "index": index})
    names = {}
    for user in users:
        name = "demo_" + user.lower()
        if name in names:
            issues.append({"kind": "username_collision", "user_ids": [names[name], user]})
        names[name] = user
    return users, issues


def save_credentials(path, rows):
    """本地凭据清单原子替换；只用于模拟账号，绝不输出密码到终端。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize(store, users, credentials, apply=False):
    """默认预览；显式执行仅补缺失账号，既有密码、停用状态不变。"""
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = {row[0] for row in db.execute("SELECT user_id FROM accounts")}
        new = sorted(set(users) - existing)
        summary = {
            "valid_users": len(users),
            "existing": len(set(users) & existing),
            "planned_new": len(new),
            "created": 0,
            "failed": 0,
        }
        if not apply:
            return summary
        rows = json.loads(credentials.read_text(encoding="utf-8")) if credentials.exists() else []
        for user in new:
            password = secrets.token_urlsafe(18)
            username = "demo_" + user.lower()
            now = time.time()
            db.execute(
                "INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?)",
                (
                    username,
                    user,
                    store.passwords.hash(password),
                    1,
                    1,
                    1,
                    now,
                    now,
                ),
            )
            rows.append(
                {
                    "username": username,
                    "password": password,
                    "user_id": user,
                    "synthetic": True,
                    **users[user],
                    "generated_at": now,
                }
            )
        if new:
            save_credentials(credentials, rows)
        summary["created"] = len(new)
        return summary


def main():
    """开发专用显式初始化/重置；预览不会创建账号，输出仅含统计。"""
    parser = argparse.ArgumentParser(description="模拟账号管理（仅本地开发）")
    parser.add_argument("action", choices=["init", "reset"])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--username")
    args = parser.parse_args()
    try:
        store = AccountStore()
        if args.action == "init":
            users, issues = inspect_users(PROJECT_ROOT / "data")
            print(json.dumps({"issues": issues, "issue_count": len(issues)}, ensure_ascii=False))
            print(json.dumps({"preview": initialize(store, users, DEFAULT_CREDENTIALS)}))
            if issues and args.apply:
                parser.error("存在数据异常，未生成账号；请先核对预览")
            print(json.dumps(initialize(store, users, DEFAULT_CREDENTIALS, args.apply)))
        else:
            if not args.username or not args.apply:
                parser.error("重置需指定--username及--apply，仅限本地模拟账号")
            row = store.get(args.username)
            if not row or not row["synthetic"]:
                parser.error("模拟账号不存在")
            rows = (
                json.loads(DEFAULT_CREDENTIALS.read_text(encoding="utf-8"))
                if DEFAULT_CREDENTIALS.exists()
                else []
            )
            rows = [x for x in rows if x["username"] != args.username]
            users, _ = inspect_users(PROJECT_ROOT / "data")
            password = secrets.token_urlsafe(18)
            store.reset(args.username, password)
            rows.append(
                {
                    "username": args.username,
                    "user_id": row["user_id"],
                    "password": password,
                    "synthetic": True,
                    **users.get(row["user_id"], {"order_count": 0, "example_order_ids": []}),
                    "generated_at": time.time(),
                }
            )
            save_credentials(DEFAULT_CREDENTIALS, rows)
            print("已重置；旧会话将失效，新演示密码仅见本地清单。")
        return 0
    except (OSError, ValueError, sqlite3.Error):
        print(json.dumps({"status": "failed", "failed": None}))
        print("账号操作失败，请检查数据或文件权限；密码及异常正文不展示。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
