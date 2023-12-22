#!/usr/bin/env python3

import os
import sys
import sqlite3
import json
import time
from pathlib import Path


DEBUG = True


def main():
    global DEBUG
    RECORDING_DB = os.getenv("MUA_RECORDING_DB")

    cur_time = time.time()

    if DEBUG:
        with open("/mua_build/exec-recorder-errors.txt", "at") as f:
            f.write(f"logging to {RECORDING_DB}\n")

    if RECORDING_DB is None:
        raise Exception("MUA_RECORDING_DB environment variable not set.")

    RECORDING_DB = Path(RECORDING_DB)
    RECORDING_DB.parent.mkdir(parents=True, exist_ok=True)

    cmd = sys.argv

    if not cmd[0].endswith("-wrap"):
        raise Exception("This script should only be used with executables ending in '-wrap'.")

    if cmd[0] == "/usr/bin/gclang-wrap":
        cmd[0] = "/root/go/bin/gclang"
    elif cmd[0] == "/usr/bin/gclang++-wrap":
        cmd[0] = "/root/go/bin/gclang++"
    else:
        raise Exception(f"Unknown executable: {cmd[0]}")

    env = os.environ.copy()

    if not RECORDING_DB.exists():
        with sqlite3.connect(RECORDING_DB) as conn:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS cmds (time, cmd, env)")
            c.close()

    cmd_json = json.dumps(cmd)
    env_json = json.dumps(env)

    with sqlite3.connect(RECORDING_DB, timeout=60) as conn:
        c = conn.cursor()
        c.execute("INSERT into cmds Values(?, ?, ?)", (cur_time, cmd_json, env_json))
        c.close()

    if True:
        with open("/mua_build/exec-recorder-errors.txt", "at") as f:
            f.write(f"logged {cur_time} {cmd_json[:20]} {env_json[:20]}\n")

    os.execl(cmd[0], *cmd)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("/mua_build/exec-recorder-errors.txt", "at") as f:
            f.write(f"{traceback.format_exc()}\n")
        raise e
