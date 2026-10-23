from importlib.resources import files

import yaml

# 从包内YAML加载提示词并导出变量，避免依赖终端当前工作目录。
_PROMPTS = yaml.safe_load(
    files("orderagent.prompts").joinpath("order_prompt.yaml").read_text(encoding="utf-8")
)


def load_prompt(name: str) -> str:
    """读取包内YAML的指定提示词，不接受外部路径。"""
    return _PROMPTS[name]


ORDER_SYSTEM_PROMPT: str = load_prompt("order_system_prompt")
