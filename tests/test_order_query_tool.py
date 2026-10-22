import json

import pytest
from pydantic import ValidationError

from orderagent.repositories.order_repository import DataLoadError, JsonOrderRepository
from orderagent.schemas.order_schema import OrderQuery, ToolContext
from orderagent.settings import Settings
from orderagent.tools import create_order_query_tool, query_order


def test_owned_order_returns_status_items_and_timestamp(order_repository, context):
    result = query_order(order_repository, OrderQuery(order_id="DEMO-HP-O-0001"), context)
    assert result.ok
    assert result.data["order_status_label"] == "待付款"
    assert result.data["items"][0]["model"]
    assert result.data["synthetic"] is True
    assert result.data["snapshot_at"].endswith("+08:00")
    assert "user_id" not in result.data


def test_unauthorized_and_absent_orders_have_same_result(order_repository, context):
    denied = query_order(order_repository, OrderQuery(order_id="DEMO-HP-O-0002"), context)
    absent = query_order(order_repository, OrderQuery(order_id="missing"), context)
    assert denied == absent
    assert denied.error.code == "ORDER_NOT_FOUND"
    assert denied.data is None


def test_explicit_trusted_context_controls_ownership(order_repository):
    args = OrderQuery(order_id="DEMO-HP-O-0002")
    assert not query_order(order_repository, args, ToolContext(user_id="DEMO-U-001")).ok
    assert query_order(order_repository, args, ToolContext(user_id="DEMO-U-002")).ok


@pytest.mark.parametrize(
    "arguments",
    [
        {"order_id": "x", "user_id": "other"},
        {"order_id": 2},
        {"order_id": ""},
        {},
        [],
        None,
    ],
)
def test_invalid_query_arguments(arguments):
    with pytest.raises(ValidationError):
        OrderQuery.model_validate(arguments)


def test_model_schema_hides_runtime_and_identity(order_repository):
    schema = create_order_query_tool(order_repository).tool_call_schema.model_json_schema()
    assert list(schema["properties"]) == ["order_id"]
    assert schema["required"] == ["order_id"]


def test_invalid_repository_data_and_duplicate_ids(tmp_path):
    rows = json.loads(Settings().orders_path.read_text(encoding="utf-8"))
    path = tmp_path / "orders.json"
    for raw in [[rows[0], rows[0]], {"wrong": "shape"}, [{"id": "missing-fields"}]]:
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(DataLoadError):
            JsonOrderRepository(path)
