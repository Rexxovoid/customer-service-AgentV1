import json
import re
from pathlib import Path

from orderagent.repositories.order_repository import DataLoadError
from orderagent.schemas.support_schema import SupportKnowledge, SupportQuery


def normalize_issue(text: str) -> str:
    """统一症状文本大小写并去除空白和常见标点，便于匹配有限别名。"""
    return re.sub(r"[\s，。！？、,.!?]", "", text.casefold())


class JsonSupportRepository:
    """按症状及别名进行保守匹配，受知识来源声明的适用条件约束。"""

    def __init__(self, paths: list[Path]):
        """读取指定知识文件，校验记录结构及跨文件ID唯一性。"""
        self._records: list[SupportKnowledge] = []
        try:
            for path in paths:
                rows = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(rows, list):
                    raise ValueError()
                self._records.extend(SupportKnowledge.model_validate(row) for row in rows)
            if len({row.id for row in self._records}) != len(self._records):
                raise ValueError()
        except (OSError, ValueError):
            raise DataLoadError("故障知识无法加载，请检查文件、字段和重复ID。") from None

    def find(self, query: SupportQuery) -> SupportKnowledge | None:
        """优先匹配上下文条件更具体的知识，无适用记录时返回None以触发交接。"""
        issue = normalize_issue(query.issue)
        environment = query.environment.casefold()
        for row in sorted(self._records, key=lambda row: len(row.required_context), reverse=True):
            if row.category != query.category:
                continue
            if not any(
                normalize_issue(alias) in issue for alias in [row.question, *row.aliases] if alias
            ):
                continue
            if row.required_context:
                applicable = all(
                    any(term.casefold() in environment for term in group)
                    for group in row.required_context
                )
            else:
                # 旧摘要没有显式条件组，按适用范围保守判断，不能视为通用方案。
                applicable = self._legacy_applicable(row.applicability, environment)
            if applicable:
                return row.model_copy(deep=True)
        return None

    @staticmethod
    def _legacy_applicable(applicability: str, environment: str) -> bool:
        """按旧知识的适用说明核对环境；无法确认型号或范围时不套用该方案。"""
        if applicability.startswith("通用低风险"):
            return True
        for brand, aliases in {
            "Windows": ("windows", "win10", "win11"),
            "Logitech": ("logitech", "罗技"),
            "Corsair": ("corsair", "海盗船"),
            "Sony": ("sony", "索尼"),
            "Surface/Microsoft": ("surface", "microsoft", "微软"),
        }.items():
            if applicability.startswith(brand):
                # 需要具体型号等证据的旧记录无法在此可靠确认，交由人工处理。
                if "型号" in applicability or "配对键" in applicability or "限" in applicability:
                    return False
                return any(alias in environment for alias in aliases)
        return False
