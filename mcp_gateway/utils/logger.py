import logging

logger = logging.getLogger("cv-mcp")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[MCP] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)