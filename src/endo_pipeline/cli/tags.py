import re

from cyclopts import App

TEST_READY = "test-ready"
GPU = "gpu"
CPU_ONLY = "cpu-only"


def get_app_tags(app: App) -> list[str]:
    return re.findall(r"#([a-z0-9\-]+)", app.help)
