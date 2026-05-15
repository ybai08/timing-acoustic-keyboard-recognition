from __future__ import annotations

from keyboard_fusion.acoustic_demo_server import AcousticDemoServer
from keyboard_fusion.web_collection_server import find_available_port


def test_find_available_port() -> None:
    port = find_available_port(0)
    assert isinstance(port, int)
    assert port > 0


def test_acoustic_demo_config_reports_missing_temp_model(tmp_path) -> None:
    server = AcousticDemoServer(("127.0.0.1", 0), model_dir=tmp_path / "missing_model")
    try:
        payload = server.config_payload()
    finally:
        server.server_close()

    assert payload["model_exists"] is False
    assert payload["model_metrics"]["session_id"] == "missing_model"
