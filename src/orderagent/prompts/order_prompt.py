from importlib.resources import files

import yaml

# 从包内YAML加载提示词并导出变量，避免依赖终端当前工作目录。
ORDER_SYSTEM_PROMPT: str = yaml.safe_load(
    files("orderagent.prompts").joinpath("order_prompt.yaml").read_text(encoding="utf-8")
)["order_system_prompt"]
