import json

import pytest
from pydantic import ValidationError

from orderagent.repositories.order_repository import DataLoadError
from orderagent.repositories.product_repository import JsonProductRepository
from orderagent.schemas.product_schema import ProductQuery
from orderagent.settings import Settings
from orderagent.tools import ProductQueryTool


@pytest.fixture(scope="module")
def repository():
    return JsonProductRepository(Settings().products_path)


@pytest.fixture(scope="module")
def product_tool(repository):
    return ProductQueryTool(repository)


@pytest.mark.parametrize("query", ["MS-P-0002", "910-004426", "罗技 M185", "罗技M185鼠标的参数"])
def test_exact_product_returns_evidence_without_invented_parameters(product_tool, query):
    result = product_tool.execute(ProductQuery(query=query))
    assert result.ok
    assert result.data["match_type"] == "detail"
    assert not result.data["needs_clarification"]
    product = result.data["products"][0]
    assert product["id"] == "MS-P-0002"
    assert product["evidence_groups"]["定位性能"]["evidence"] == [
        "Optical technology",
        "20 ips speed",
    ]
    assert "dpi" not in product["description"].lower()
    assert product["source_last_updated"] == "2016-05-09"
    assert product["current_price"] is None
    assert product["stock"] is None
    assert "historical_price" not in product
    assert "original_fields" not in product
    assert product["source"]["url"]


@pytest.mark.parametrize(
    "query,category,total",
    [
        ("鼠标", "mice", 90),
        ("mouse", "mice", 90),
        ("键盘", "keyboards", 89),
        ("耳机", "headphones", 200),
    ],
)
def test_category_query_returns_bounded_candidates(product_tool, query, category, total):
    result = product_tool.execute(ProductQuery(query=query))
    assert result.ok
    assert result.data["match_type"] == "candidates"
    assert result.data["needs_clarification"]
    assert result.data["total_matches"] == total
    assert result.data["returned_count"] == 5
    assert {p["category"] for p in result.data["products"]} == {category}
    assert len(result.data["products"]) == 5
    assert all(p["description"] for p in result.data["products"])


def test_brand_and_category_query_does_not_mix_categories(product_tool):
    result = product_tool.execute(ProductQuery(query="罗技 鼠标"))
    assert result.ok
    assert all(
        p["brand"] == "Logitech" and p["category"] == "mice" for p in result.data["products"]
    )


@pytest.mark.parametrize("query", ["不存在的商品XYZ999", "参数", "鼠标 键盘"])
def test_no_match_is_explicit(product_tool, query):
    result = product_tool.execute(ProductQuery(query=query))
    assert not result.ok
    assert result.error.code == "PRODUCT_NOT_FOUND"
    assert result.data is None


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"query": ""},
        {"query": "  "},
        {"query": 42},
        {"query": "x" * 201},
        {"query": "鼠标", "user_id": "other"},
    ],
)
def test_query_validation(value):
    with pytest.raises(ValidationError):
        ProductQuery.model_validate(value)


def test_model_facing_schema_has_only_query(product_tool):
    tool = product_tool.as_langchain_tool()
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]
    # LangChain rebuilds the model-facing subset; the execution schema stays strict.
    assert tool.args_schema.model_json_schema()["additionalProperties"] is False
    with pytest.raises(ValidationError):
        tool.invoke({"query": "鼠标", "user_id": "other"})


def test_results_are_detached_from_repository(repository):
    found = repository.search("MS-P-0002")
    found[0].evidence_groups.clear()
    assert repository.search("MS-P-0002")[0].evidence_groups


@pytest.mark.parametrize("raw", [{"wrong": "shape"}, [{"id": "missing-fields"}], None])
def test_malformed_product_data_is_rejected(tmp_path, raw):
    path = tmp_path / "products.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataLoadError, match="商品数据"):
        JsonProductRepository(path)


def test_duplicate_product_id_is_rejected(tmp_path):
    product = json.loads(Settings().products_path.read_text(encoding="utf-8"))[0]
    path = tmp_path / "products.json"
    path.write_text(json.dumps([product, product]), encoding="utf-8")
    with pytest.raises(DataLoadError):
        JsonProductRepository(path)


def test_product_data_path_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("PRODUCT_DATA_PATH", raising=False)
    env = tmp_path / ".env"
    configured = tmp_path / "catalog.json"
    env.write_text(f"PRODUCT_DATA_PATH={configured.as_posix()}\n", encoding="utf-8")
    assert Settings.load(env).products_path == configured
    override = tmp_path / "override.json"
    monkeypatch.setenv("PRODUCT_DATA_PATH", str(override))
    assert Settings.load(env).products_path == override
