"""本地聊天页面接入：HTTP会话连接现有官方Agent图，不重新定义业务规则。"""

import argparse
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from orderagent.agents.agent_factory import create_order_agent
from orderagent.evaluation.runner import Collector, flatten_runs
from orderagent.observability import get_observability
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import PROJECT_ROOT, Settings

STATIC_DIR = Path(__file__).with_name("static")
COOKIE_NAME = "orderagent_session"


class ChatRequest(BaseModel):
    """页面仅提交消息，不接受客户端指定身份、线程或业务终态。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=4000)


@dataclass
class BrowserSession:
    """服务端会话信息；锁覆盖聊天、读取和重置，避免同线程交叉更新。"""

    thread_id: str = field(default_factory=lambda: uuid4().hex)
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    log_id: str = field(default_factory=lambda: uuid4().hex)
    lock: Lock = field(default_factory=Lock)

    @property
    def config(self):
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 40}


def create_web_app(settings: Settings, user_id: str, *, graph=None, log_dir=None) -> FastAPI:
    """固定演示用户；通过官方invoke/checkpointer执行及重置浏览器会话。"""
    context = ToolContext(user_id=user_id)
    agent = graph if graph is not None else create_order_agent(settings)
    observability = get_observability(settings)
    sessions: dict[str, BrowserSession] = {}
    registry_lock = Lock()
    log_dir = Path(log_dir) if log_dir is not None else PROJECT_ROOT / "runtime" / "logs"
    log_lock = Lock()
    log_secrets = [
        settings.api_key.get_secret_value(),
        settings.ragflow_api_key.get_secret_value(),
        settings.qiwen_api_key.get_secret_value(),
    ]

    def write_record(name, record):
        """每请求一份格式化JSON，原子替换；保留每类20MiB额度，失败不改变业务。"""
        temporary = None
        try:
            text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
            for secret in log_secrets:
                if secret:
                    text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
            with log_lock:
                directory = log_dir / name
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"{record['request_id']}.json"
                temporary = directory / f".{uuid4().hex}.tmp"
                temporary.write_text(text, encoding="utf-8")
                temporary.replace(target)
                # 只清理本模块生成的编号JSON；旧JSONL及无关文件均不动。
                files = sorted(
                    (p for p in directory.glob("*.json") if re.fullmatch(r"[a-f0-9]{32}", p.stem)),
                    key=lambda p: p.stat().st_mtime_ns,
                )
                total = sum(p.stat().st_size for p in files)
                for old in files:
                    if total <= 20 * 1024 * 1024:
                        break
                    if old != target:
                        size = old.stat().st_size
                        old.unlink()
                        total -= size
        except Exception:
            print("Web本地日志写入失败；请求结果不受影响。")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def error_metadata(exc):
        """异常仅记录类型和分类，避免任意异常正文带入用户消息/密钥/模型正文。"""
        cause = exc.__cause__ or exc.__context__
        root = exc
        seen = set()
        while id(root) not in seen:
            seen.add(id(root))
            next_cause = root.__cause__ or root.__context__
            if next_cause is None:
                break
            root = next_cause
        name = type(root).__name__
        return {
            "error_type": type(exc).__name__,
            "cause_type": type(cause).__name__ if cause is not None else None,
            "root_cause_type": name,
            "cause_category": "timeout" if "Timeout" in name else "execution_error",
        }

    app = FastAPI(
        title="OrderAgent 本地聊天",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        request.state.error = {
            "error_type": "RequestValidationError",
            "cause_type": None,
            "root_cause_type": None,
            "cause_category": "invalid_input",
        }
        return JSONResponse(
            {
                "detail": "请输入1—4000字符的消息，请勿附加身份或线程参数。",
                "request_id": request.state.request_id,
            },
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        if not getattr(request.state, "error", None):
            request.state.error = {
                "error_type": type(exc).__name__,
                "cause_type": None,
                "root_cause_type": None,
                "cause_category": "http_rejected",
            }
        return JSONResponse(
            {"detail": exc.detail, "request_id": request.state.request_id},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.middleware("http")
    async def response_headers(request, call_next):
        request.state.request_id = uuid4().hex
        request.state.log_session = None
        started = time.perf_counter()
        created_at = datetime.now(timezone.utc).isoformat()
        status = None
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:
            request.state.error = error_metadata(exc)
            status = 500
            response = JSONResponse(
                {
                    "detail": "服务暂时无法完成请求，请保留问题编号。",
                    "request_id": request.state.request_id,
                },
                status_code=500,
            )
        finally:
            if request.url.path.startswith("/api/"):
                write_record(
                    "requests",
                    {
                        "request_id": request.state.request_id,
                        "created_at": created_at,
                        "session_id": request.state.log_session,
                        "method": request.method,
                        "endpoint": request.url.path
                        if request.url.path in {"/api/chat", "/api/reset", "/api/session"}
                        else "/api/other",
                        "seconds": time.perf_counter() - started,
                        "http_status": status,
                        **getattr(
                            request.state,
                            "error",
                            {
                                "error_type": None,
                                "cause_type": None,
                                "root_cause_type": None,
                                "cause_category": None,
                            },
                        ),
                    },
                )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    def current_session(request: Request) -> BrowserSession:
        with registry_lock:
            session = sessions.get(request.cookies.get(COOKIE_NAME, ""))
        if session is None:
            raise HTTPException(401, "会话已失效，请刷新页面重新连接。")
        request.state.log_session = session.log_id
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "请从本地聊天页面发起请求。")
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "请求来源不匹配。")
        if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), session.csrf):
            raise HTTPException(403, "会话校验失败，请刷新页面。")
        return session

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/session")
    def session_info(request: Request, response: Response):
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "请从本地聊天页面发起请求。")
        with registry_lock:
            token = request.cookies.get(COOKIE_NAME, "")
            session = sessions.get(token)
            if session is None:
                if len(sessions) >= 128:
                    raise HTTPException(503, "本地演示会话已满，请重启服务后再试。")
                token = secrets.token_urlsafe(32)
                session = BrowserSession()
                sessions[token] = session
        request.state.log_session = session.log_id
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "该会话仍在处理中，请稍后刷新。")
        try:
            state = agent.get_state(session.config).values
            messages = [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "text": m.text}
                for m in state.get("messages", [])
                if isinstance(m, HumanMessage)
                or (isinstance(m, AIMessage) and not m.tool_calls and m.text)
            ]
            response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict")
            return {"user_id": context.user_id, "csrf_token": session.csrf, "messages": messages}
        finally:
            session.lock.release()

    @app.post("/api/chat")
    def chat(
        body: ChatRequest, request: Request, session: BrowserSession = Depends(current_session)
    ):
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "上一条消息仍在处理中，请勿重复发送。")
        collector = Collector()
        result = None
        try:
            with (
                tracing_context(enabled=False),
                observability.request(
                    source="web_chat",
                    session_id=session.log_id,
                    request_id=request.state.request_id,
                    knowledge_source="configured_rag" if settings.ragflow_api_url else "local_json",
                ) as tracking,
            ):
                try:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": body.message}]},
                        config={
                            **session.config,
                            **tracking,
                            "callbacks": [collector, *tracking.get("callbacks", [])],
                        },
                        context=context,
                    )
                finally:
                    observability.diagnostics(collector.events, result, collector.traced_runs)
            return {"answer": result["messages"][-1].text, "request_id": request.state.request_id}
        except Exception as exc:
            request.state.error = error_metadata(exc)
            # 不重试整轮，避免已执行工具后重复办理；底层SDK仍遵循原有配置。
            raise HTTPException(
                502, "本轮未能完整返回。操作结果可能已保存，请先查询进度，勿重复确认退货。"
            ) from None
        finally:
            try:
                steps = []
                for run in flatten_runs(collector.traced_runs):
                    item = {k: run[k] for k in ("id", "parent_id", "name", "type", "seconds")}
                    # 官方trace错误可能带正文/堆栈，只提取异常类名。
                    error = re.search(r"\b(\w+(?:Error|Exception))\b", run["error"] or "")
                    item["error_type"] = error.group(1) if error else None
                    output = (run["outputs"] or {}).get("output")
                    artifact = getattr(output, "artifact", None)
                    if artifact is None and isinstance(getattr(output, "update", None), dict):
                        artifacts = [
                            getattr(m, "artifact", None) for m in output.update.get("messages", [])
                        ]
                        artifact = next((a for a in artifacts if isinstance(a, dict)), None)
                    if isinstance(artifact, dict):
                        data = artifact.get("data") or {}
                        item.update(
                            ok=artifact.get("ok"),
                            code=(artifact.get("error") or {}).get("code") or data.get("code"),
                            action=data.get("action"),
                        )
                        if data.get("id") and data.get("data_origin"):
                            item["knowledge"] = {
                                "id": data["id"],
                                "data_origin": data["data_origin"],
                            }
                    steps.append(item)
                rag = []
                for event in collector.events:
                    if event["name"] != "rag_diagnostic":
                        continue
                    data = event["data"]
                    rag.append(
                        {
                            "elapsed_seconds": event["elapsed_seconds"],
                            **{
                                k: data[k]
                                for k in (
                                    "stage",
                                    "error_code",
                                    "error_type",
                                    "failure_reasons",
                                    "seconds",
                                    "debug_status",
                                    "debug_record",
                                    "chunk_id",
                                    "document",
                                    "candidate_count",
                                    "completed",
                                    "raw_candidate_count",
                                    "nonempty_candidate_count",
                                    "filtered_candidate_count",
                                    "handling",
                                    "applicable",
                                    "excerpt_valid",
                                )
                                if k in data
                            },
                        }
                    )
                    if isinstance(data.get("selection"), dict):
                        rag[-1]["selection"] = {
                            k: data["selection"][k]
                            for k in ("chunk_id", "handling", "applicable", "decision")
                            if k in data["selection"]
                        }
                    if isinstance(data.get("candidates"), list):
                        rag[-1]["sources"] = [
                            {
                                k: candidate[k]
                                for k in ("id", "document", "similarity")
                                if k in candidate
                            }
                            for candidate in data["candidates"][:5]
                        ]
                write_record(
                    "agent",
                    {
                        "request_id": request.state.request_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "session_id": session.log_id,
                        "steps": steps,
                        "rag_diagnostics": rag,
                        "state": {
                            k: result.get(k)
                            for k in (
                                "active_support_topic",
                                "support_phase",
                                "return_stage",
                                "return_error_code",
                                "support_question_count",
                                "return_request_count",
                                "return_fact_diagnostics",
                                "return_application_revision",
                                "return_reason_update_pending",
                                "return_information_pending",
                            )
                        }
                        if result is not None
                        else None,
                        "error": getattr(request.state, "error", None),
                    },
                )
            except Exception:
                print("Agent日志摘要生成失败；请求结果不受影响。")
            finally:
                session.lock.release()

    @app.post("/api/reset")
    def reset(session: BrowserSession = Depends(current_session)):
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "消息仍在处理中，暂时不能新建会话。")
        try:
            agent.checkpointer.delete_thread(session.thread_id)
            session.thread_id = uuid4().hex
            return {"ok": True}
        finally:
            session.lock.release()

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> int:
    """启动单进程、仅本机可访问的演示服务，不开放生产身份认证。"""
    parser = argparse.ArgumentParser(description="OrderAgent 本地聊天页面")
    parser.add_argument("--user-id", required=True, help="服务端固定的演示用户，如 DEMO-U-004")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port须在1—65535之间")
    try:
        app = create_web_app(Settings.load(), args.user_id)
    except Exception:
        print("启动失败，请检查用户ID、数据路径及模型/RAG配置；敏感详情不会展示。")
        return 1
    print(f"聊天页面：http://127.0.0.1:{args.port}（人工与退货均为模拟）")
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
