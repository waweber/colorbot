# Logging
import logging

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(handler)

colorbot_logger = logging.getLogger("colorbot")

tensorflow_logger = logging.getLogger("tensorflow")
tensorflow_logger.setLevel(logging.WARNING)
