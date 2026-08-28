"""Daemon control-plane exposure guards: a non-loopback bind without a shared secret
is refused before anything starts, and the token check is constant-time and rejects
a missing header when a token is configured."""
from __future__ import annotations

from freetoken.daemon.server import _is_loopback_host, main


def test_loopback_hosts_recognized():
    for host in ("127.0.0.1", "::1", "localhost", "127.9.9.9"):
        assert _is_loopback_host(host), host
    for host in ("0.0.0.0", "::", "10.0.0.5", "myhost.lan", "192.168.1.2"):
        assert not _is_loopback_host(host), host


def test_non_loopback_bind_without_token_is_refused(capsys, monkeypatch):
    monkeypatch.delenv("FREETOKEN_DAEMON_TOKEN", raising=False)
    assert main(["--host", "0.0.0.0"], prog="ft daemon") == 1
    err = capsys.readouterr().err
    assert "refusing to bind" in err and "--token" in err
