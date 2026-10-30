import importlib.util
import sys
from config import MODE

def get_tool_env(mode: str = "train"):
    """Return the ToolEnv class from the requested split."""
    mode = mode.lower()
    if mode not in MODE:
        raise ValueError(f"Invalid mode: {mode!r}. Only supports: {list(MODE.keys())}")

    tools_dir = MODE[mode]["tools_dir"]
    module_name = f"agent_tools_{mode}"
    if module_name in sys.modules:
        return sys.modules[module_name].ToolEnv

    zaic_init = tools_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(zaic_init),
        submodule_search_locations=[str(zaic_init.parent)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.ToolEnv

__all__ = ["get_tool_env"]
