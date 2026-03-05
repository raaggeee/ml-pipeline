import logging

logger = logging.getLogger("test_logger")
logger.setLevel("DEBUG")
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.formatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler()
file_handler.setLevel("ERROR")
file_handler.formatter(formatter)
logger.addHandler(file_handler)
