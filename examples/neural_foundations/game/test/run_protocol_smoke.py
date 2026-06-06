#!/usr/bin/env python3
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


HOST = "127.0.0.1"
PORT = 11008
GAME_ROOT = Path(__file__).resolve().parents[1]
SCENE = "res://unit_03_racer/racer_train.tscn"
GODOT = os.environ.get("GODOT_BIN", os.environ.get("GODOT", "godot"))


def recvall(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("socket closed early")
        data += chunk
    return data


def send(sock: socket.socket, message: dict) -> None:
    data = json.dumps(message).encode("utf-8")
    sock.sendall(len(data).to_bytes(4, "little") + data)


def recv(sock: socket.socket) -> dict:
    size = int.from_bytes(recvall(sock, 4), "little")
    return json.loads(recvall(sock, size).decode("utf-8"))


def main() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(30)
    process = None
    connection = None
    failures: list[str] = []

    try:
        process = subprocess.Popen(
            [
                GODOT,
                "--headless",
                "--path",
                str(GAME_ROOT),
                SCENE,
                "action_repeat=1",
                "speedup=1",
                "read_timeout=10",
                "connect_timeout=10",
            ]
        )
        connection, _address = server.accept()
        connection.settimeout(30)

        send(connection, {"type": "handshake", "major_version": "0", "minor_version": "3"})
        send(connection, {"type": "env_info"})
        env_info = recv(connection)
        if env_info.get("type") != "env_info":
            failures.append("env_info type")
        if env_info.get("n_agents") != 1:
            failures.append("n_agents")
        if env_info.get("action_space", {}).get("drive", {}).get("size") != 2:
            failures.append("two-value continuous action")

        send(connection, {"type": "reset"})
        reset = recv(connection)
        if reset.get("type") != "reset":
            failures.append("reset type")
        if len(reset.get("obs") or []) != 1:
            failures.append("reset observation count")

        saw_step = False
        saw_terminal = False
        for _step in range(120):
            send(connection, {"type": "action", "action": [{"drive": [0.0, 0.0]}]})
            step = recv(connection)
            if step.get("type") != "step":
                failures.append("step type")
                break
            saw_step = True
            done = step.get("done") or []
            if done and done[0]:
                saw_terminal = True
                send(connection, {"type": "reset"})
                second_reset = recv(connection)
                if second_reset.get("type") != "reset":
                    failures.append("second reset type")
                break

        if not saw_step:
            failures.append("one observation/reward step")
        if not saw_terminal:
            failures.append("terminal episode")

        send(connection, {"type": "close"})
    finally:
        if process is not None:
            try:
                return_code = process.wait(timeout=15)
            except Exception:
                process.kill()
                return_code = -1
            if return_code != 0:
                failures.append(f"godot exited with code {return_code}")
        if connection is not None:
            connection.close()
        server.close()

    if failures:
        print("protocol smoke: FAIL", failures)
        sys.exit(1)
    print("protocol smoke: PASS")


if __name__ == "__main__":
    main()
