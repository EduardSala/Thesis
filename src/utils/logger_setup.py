import logging


def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format= '%(levelname)s - %(asctime)s - %(message)s',
        handlers=[logging.FileHandler("thesis.log", mode="w"), logging.StreamHandler()]
    )
    return logging.getLogger("SatelliteLog")


logger = setup_logger()

