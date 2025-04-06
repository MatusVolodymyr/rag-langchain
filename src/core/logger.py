import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a",  # 'a' for appending;'w' to overwrite on each run
)

logger = logging.getLogger("RAG_API")
