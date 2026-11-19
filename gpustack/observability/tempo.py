import logging
import os
import sys
import time

import yaml

from gpustack.utils.network import is_port_available, is_port_reusable

logger = logging.getLogger(__name__)

_TEMPO_BINARY = "/usr/local/bin/tempo"
_PORT_RETRY_INTERVAL = 5


def run(config_path: str) -> None:
    """Wait for the configured listeners before handing the s6 service to Tempo."""
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    server = config["server"]
    receiver = config["distributor"]["receivers"]["otlp"]["protocols"]["grpc"]
    ports = (
        int(receiver["endpoint"].rsplit(":", 1)[1]),
        server["http_listen_port"],
        server["grpc_listen_port"],
    )

    waiting_for = ()
    while True:
        busy = tuple(
            port
            for port in ports
            if not is_port_available(port) and not is_port_reusable(port)
        )
        if not busy:
            break
        if busy != waiting_for:
            logger.warning("Built-in Tempo is waiting for busy TCP ports %s", busy)
            waiting_for = busy
        time.sleep(_PORT_RETRY_INTERVAL)

    if waiting_for:
        logger.info("Built-in Tempo ports are available. Starting Tempo.")
    os.execv(_TEMPO_BINARY, [_TEMPO_BINARY, f"-config.file={config_path}"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(sys.argv[1])
