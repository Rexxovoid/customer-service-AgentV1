import pytest

from orderagent.repositories import JsonOrderRepository
from orderagent.schemas.order_schema import ToolContext
from orderagent.settings import Settings
from orderagent.tools import OrderQueryTool


@pytest.fixture
def order_tool():
    return OrderQueryTool(JsonOrderRepository(Settings().orders_path))


@pytest.fixture
def context():
    return ToolContext(user_id="DEMO-U-001")
