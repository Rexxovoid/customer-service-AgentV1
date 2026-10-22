from importlib.resources import files

import yaml

ORDER_SYSTEM_PROMPT: str = yaml.safe_load(
    files("orderagent.prompts").joinpath("order_prompt.yaml").read_text(encoding="utf-8")
)["order_system_prompt"]
