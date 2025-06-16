# utils/logger.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("[APP]")
tool_logger = logging.getLogger("[TOOL]")
agent_logger = logging.getLogger("[AGENT]")
