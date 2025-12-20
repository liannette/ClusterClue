import logging
import logging.handlers

# I will be honest - I do not understand logging in multiprocessing at all.
# This code has been written by trial and error, and there might be better ways to do this.


def setup_logging(log_filepath, verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    formatting = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if verbose else '%(asctime)s - %(levelname)s - %(message)s'
    
    # Suppress most external logs by default
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    ipresto_logger = logging.getLogger("ipresto")
    ipresto_logger.setLevel(level)

    # Clear any existing handlers
    if ipresto_logger.hasHandlers():
        ipresto_logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(formatting))
    file_handler.setLevel(level)
    ipresto_logger.addHandler(file_handler)

    # Disable propagation to avoid duplicate logs in root logger
    ipresto_logger.propagate = False


def listener_configurer(log_filepath, verbose):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if verbose else
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    

def listener_process(queue, log_filepath, verbose):
    listener_configurer(log_filepath, verbose)

    while True:
        record = queue.get()
        if record is None:  # sentinel
            break

        logger = logging.getLogger(record.name)
        logger.handle(record)


def worker_configurer(queue, verbose=False):
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers = []
    root.addHandler(handler)


def worker_init(q):
    global log_queue
    log_queue = q
    worker_configurer(log_queue)