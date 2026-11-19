import logging
from unittest.mock import Mock

import yaml

from gpustack.observability import tempo


def test_tempo_waits_until_listeners_are_available(tmp_path, monkeypatch, caplog):
    config_path = tmp_path / "tempo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "server": {"http_listen_port": 3200, "grpc_listen_port": 9095},
                "distributor": {
                    "receivers": {
                        "otlp": {"protocols": {"grpc": {"endpoint": "0.0.0.0:4317"}}}
                    }
                },
            }
        )
    )
    busy = {4317, 9095}
    launch = Mock()
    delays = []

    def release_port_on_retry(delay):
        launch.assert_not_called()
        delays.append(delay)
        busy.remove(max(busy))

    monkeypatch.setattr(tempo, "is_port_available", lambda port: port not in busy)
    monkeypatch.setattr(tempo, "is_port_reusable", lambda port: False)
    monkeypatch.setattr(tempo.time, "sleep", release_port_on_retry)
    monkeypatch.setattr(tempo.os, "execv", launch)

    with caplog.at_level(logging.INFO):
        tempo.run(str(config_path))

    assert delays == [5, 5]
    assert "waiting for busy TCP ports" in caplog.text
    assert "Starting Tempo" in caplog.text
    launch.assert_called_once_with(
        "/usr/local/bin/tempo",
        ["/usr/local/bin/tempo", f"-config.file={config_path}"],
    )
