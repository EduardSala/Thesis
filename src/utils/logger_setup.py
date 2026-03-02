import logging
import os


def setup_logger():

    log = logging.getLogger("SatelliteLog")
    log.setLevel(logging.DEBUG)

    if not log.handlers:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        file_handler = logging.FileHandler(os.path.join(log_dir, "thesis.log"),mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(levelname)s: %(message)s')
        file_handler.setFormatter(file_format)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_format = logging.Formatter('%(asctime)s --- %(levelname)s: %(message)s')
        stream_handler.setFormatter(stream_format)

        log.addHandler(file_handler)
        log.addHandler(stream_handler)

    return log


logger = setup_logger()


