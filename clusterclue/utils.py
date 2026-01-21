import logging
import logging.handlers

# I will be honest - I do not understand logging in multiprocessing at all.
# This code has been written by trial and error, and there might be better ways to do this.


def listener_configurer(log_filepath, verbose=False):
    """
    Configure the listener process logger (writes to file).
    """
    root = logging.getLogger()
    root.handlers.clear()

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if verbose else
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Third-party libs default to WARNING
    root.setLevel(logging.WARNING)

    # clusterclue logs are INFO/DEBUG
    clusterclue_logger = logging.getLogger("clusterclue")
    clusterclue_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    clusterclue_logger.propagate = True


def listener_process(queue, log_filepath, verbose=False):
    """
    Listener process: consumes log records from queue and writes to file.
    """
    listener_configurer(log_filepath, verbose)

    while True:
        record = queue.get()
        if record is None:  # sentinel to stop listener
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def worker_configurer(queue, verbose=False):
    """I don't decide anything, I just forward"""
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.DEBUG)  # forward everything
    root.addHandler(handler)


def worker_init(q):
    global log_queue
    log_queue = q
    worker_configurer(log_queue)