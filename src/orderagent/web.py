"""本地聊天页面接入：HTTP会话连接现有官方Agent图，不重新定义业务规则。"""

import argparse
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from orderagent.agents.agent_factory import create_order_agent
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import Settings

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
    lock: Lock = field(default_factory=Lock)

    @property
    def config(self):
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 40}


def create_web_app(settings: Settings, user_id: str, *, graph=None) -> FastAPI:
    """固定演示用户；通过官方invoke/checkpointer执行及重置浏览器会话。"""
    context = ToolContext(user_id=user_id)
    agent = graph if graph is not None else create_order_agent(settings)
    sessions: dict[str, BrowserSession] = {}
    registry_lock = Lock()
    app = FastAPI(title="OrderAgent 本地聊天", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        return JSONResponse(
            {"detail": "请输入1—4000字符的消息，请勿附加身份或线程参数。"}, status_code=422
        )

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
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
    def chat(body: ChatRequest, session: BrowserSession = Depends(current_session)):
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "上一条消息仍在处理中，请勿重复发送。")
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": body.message}]},
                config=session.config,
                context=context,
            )
            return {"answer": result["messages"][-1].text}
        except Exception:
            # 不重试整轮，避免已执行工具后重复办理；底层SDK仍遵循原有配置。
            raise HTTPException(
                502, "本轮未能完整返回。操作结果可能已保存，请先查询进度，勿重复确认退货。"
            ) from None
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
