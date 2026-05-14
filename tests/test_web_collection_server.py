from __future__ import annotations

from keyboard_fusion.web_collection_server import find_available_port


def test_find_available_port() -> None:
    port = find_available_port(0)
    assert isinstance(port, int)
    assert port > 0

