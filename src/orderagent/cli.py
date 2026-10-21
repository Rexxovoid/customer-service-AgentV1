import argparse
import sys

from pydantic import ValidationError

from orderagent.agents.agent_factory import create_order_agent
from orderagent.agents.order_agent import OrderAgent
from orderagent.exceptions import AgentExecutionError
from orderagent.repositories.order_repository import DataLoadError, JsonOrderRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import ConfigurationError, Settings
from orderagent.tools import OrderQueryTool


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="DeepSeek 订单查询 Agent（本地演示）")
    parser.add_argument("--user-id", required=True, help="应用提供的演示用户，如 DEMO-U-001")
    parser.add_argument("--message", help="单轮自然语言查询；省略则进入交互会话")
    parser.add_argument("--order-id", help="直接调用查询工具，不调用模型且无需API密钥")
    parser.add_argument("--trace", action="store_true", help="显示结构化工具执行结果")
    args = parser.parse_args()
    if args.order_id and args.message:
        parser.error("--order-id 与 --message 不能同时使用")
    try:
        settings = Settings.load()
        context = ToolContext(user_id=args.user_id)
        if args.order_id:
            from orderagent.schemas.order_schema import OrderQuery

            tool = OrderQueryTool(JsonOrderRepository(settings.orders_path))
            result = tool.execute(OrderQuery(order_id=args.order_id), context)
            print(result.model_dump_json(indent=2))
            return 0 if result.ok else 1
        graph = create_order_agent(settings)
        agent = OrderAgent(graph, context)

        def ask(text: str) -> None:
            reply = agent.chat(text)
            print(reply.model_dump_json(indent=2) if args.trace else reply.answer)

        if args.message:
            ask(args.message)
            return 0
        print("订单助手已启动。输入 /exit 退出，/reset 清空会话。当前为模拟订单数据。")
        while True:
            try:
                text = input("你 > ").strip()
                if text == "/exit":
                    return 0
                if text == "/reset":
                    agent.reset()
                    print("会话已清空。")
                elif text:
                    try:
                        ask(text)
                    except (AgentExecutionError, ValueError) as exc:
                        print(f"查询失败：{exc}")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
    except (ConfigurationError, DataLoadError, AgentExecutionError) as exc:
        print(f"错误：{exc}")
        return 1
    except ValidationError:
        print("错误：用户ID或订单号格式不合法。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
