import argparse
import sys
from uuid import uuid4

from pydantic import ValidationError

from orderagent.agents.agent_factory import create_order_agent
from orderagent.repositories.after_sales_repository import AfterSalesRepository
from orderagent.repositories.order_repository import DataLoadError, JsonOrderRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import ConfigurationError, Settings
from orderagent.tools import query_order


def main() -> int:
    """解析命令参数，运行直接订单查询或Agent会话，并返回进程退出码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="DeepSeek 数码售后 Agent（本地演示）")
    parser.add_argument("--user-id", required=True, help="应用提供的演示用户，如 DEMO-U-001")
    parser.add_argument("--message", help="单轮自然语言查询；省略则进入交互会话")
    parser.add_argument("--order-id", help="直接调用查询工具，不调用模型且无需API密钥")
    parser.add_argument("--trace", action="store_true", help="显示框架原生执行更新与工具消息")
    args = parser.parse_args()
    if args.order_id and args.message:
        parser.error("--order-id 与 --message 不能同时使用")
    if args.message is not None and (not args.message.strip() or len(args.message) > 4000):
        parser.error("请输入1—4000字符的消息。")
    try:
        settings = Settings.load()
        context = ToolContext(user_id=args.user_id)
        if args.order_id:
            from orderagent.schemas.order_schema import OrderQuery

            result = query_order(
                JsonOrderRepository(
                    settings.orders_path, AfterSalesRepository(settings.after_sales_db_path)
                ),
                OrderQuery(order_id=args.order_id),
                context,
            )
            print(result.model_dump_json(indent=2))
            return 0 if result.ok else 1
        graph = create_order_agent(settings)
        config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 40}

        def ask(text: str) -> None:
            """在当前线程提交一轮消息，按需显示执行更新并打印最终回答。"""
            if not text.strip() or len(text) > 4000:
                print("请输入1—4000字符的消息。")
                return
            result = graph.invoke(
                {"messages": [{"role": "user", "content": text.strip()}]},
                config=config,
                context=context,
                print_mode="updates" if args.trace else (),
            )
            print(result["messages"][-1].text)

        if args.message:
            ask(args.message)
            return 0
        print("数码售后助手已启动。/exit退出，/reset清空会话。订单、退货入口和人工交接均为模拟。")
        while True:
            try:
                text = input("你 > ").strip()
                if text == "/exit":
                    return 0
                if text == "/reset":
                    graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
                    config["configurable"]["thread_id"] = uuid4().hex
                    print("会话已清空。")
                elif text:
                    try:
                        ask(text)
                    except Exception:
                        print("查询失败，请检查模型配置或稍后重试。可输入 /reset 开始新会话。")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
    except (ConfigurationError, DataLoadError) as exc:
        print(f"错误：{exc}")
        return 1
    except ValidationError:
        print("错误：用户ID或订单号格式不合法。")
        return 1
    except Exception:
        print("执行失败，请检查模型配置、调用限制或稍后重试。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
