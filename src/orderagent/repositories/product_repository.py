import json
import re
from pathlib import Path

from orderagent.repositories.order_repository import DataLoadError
from orderagent.schemas.product_schema import StoredProduct

BRAND_ALIASES = {"罗技": "logitech", "雷蛇": "razer", "海盗船": "corsair", "索尼": "sony"}
CATEGORY_ALIASES = {
    "鼠标": "mice",
    "mouse": "mice",
    "mice": "mice",
    "键盘": "keyboards",
    "keyboard": "keyboards",
    "keyboards": "keyboards",
    "耳机": "headphones",
    "headphone": "headphones",
    "headphones": "headphones",
    "headset": "headphones",
    "headsets": "headphones",
    "earbuds": "headphones",
}


def normalize(text: str) -> str:
    """统一大小写并提取中英文及数字词元，供确定性的关键词匹配使用。"""
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold()))


class JsonProductRepository:
    """小型只读商品目录，先精确匹配标识，再匹配别名和关键词。"""

    def __init__(self, path: Path):
        """加载商品记录并校验结构及ID唯一性，异常时返回脱敏错误。"""
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, list):
                raise ValueError()
            self._products = [StoredProduct.model_validate(row) for row in raw]
            if len({row.id.casefold() for row in self._products}) != len(self._products):
                raise ValueError()
        except (OSError, ValueError):
            raise DataLoadError("商品数据无法加载，请检查文件格式、必需字段与重复ID。") from None

    def search(self, query: str) -> list[StoredProduct]:
        """优先查ID、型号或源SKU，再按品类别名及关键词筛选并返回记录深拷贝。"""
        exact = [
            row
            for row in self._products
            if query.strip().casefold()
            in {
                row.id.casefold(),
                row.model.casefold(),
                row.source_sku.casefold(),
            }
        ]
        if exact:
            return [row.model_copy(deep=True) for row in exact]
        text = query.casefold()
        for alias, brand in BRAND_ALIASES.items():
            text = text.replace(alias, f" {brand} ")
        for alias in ("鼠标", "键盘", "耳机"):
            text = text.replace(alias, f" {CATEGORY_ALIASES[alias]} ")
        for word in ("具体参数", "性能信息", "商品信息", "参数", "性能", "介绍", "查询", "的"):
            text = text.replace(word, " ")
        terms = set(normalize(text).split())
        if not terms:
            return []
        categories = {CATEGORY_ALIASES[t] for t in terms if t in CATEGORY_ALIASES}
        if len(categories) > 1:
            return []
        terms -= CATEGORY_ALIASES.keys()
        matches = []
        for row in self._products:
            if categories and row.category not in categories:
                continue
            searchable = normalize(" ".join([row.id, row.brand, row.name, row.model]))
            if terms <= set(searchable.split()):
                matches.append(row.model_copy(deep=True))
        return matches
