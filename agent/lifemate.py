from pathlib import Path
from config import MODE
from models import get_asr, get_llm
from prompt.builder import LifeMatePromptBuilder
from agent.tools import get_tool_env

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class LifeMateAgent:
  """LifeMate Agentic Assistant."""

  def __init__(
      self,
      asr: str ="gipformer",
      llm: str = "Qwen/Qwen3.5-4B",
      mode: str = "train",
  ):
    self.mode = mode

    # Load config
    cfg = MODE.get(self.mode)
    assert cfg, f"Mode {self.mode} not found in config"
    
    self.tools_dir = Path(cfg.get("tools_dir")).resolve()
    self.catalog_path = Path(cfg.get("catalog_path")).resolve()
    self.data_dir = Path(cfg.get("data_dir")).resolve()

    # Init ASR
    logger.info(f"Initialize ASR: {asr}")
    self.asr = get_asr(asr)
    
    # Init LLM
    logger.info(f"Initialize LLM: {llm}")
    # self.llm = get_llm(model_id=llm)
    
    # Load ToolEnv
    self.ToolEnv = get_tool_env(self.mode)

    # Build Prompt
    self.prompt_builder = LifeMatePromptBuilder(self.catalog_path)

    logger.info(f"LifeMate Agent [{self.mode.upper()}] is ready!")

if __name__ == "__main__":
  agent = LifeMateAgent()