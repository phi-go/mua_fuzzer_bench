#!/usr/bin/env python3

import os
import sys
import sqlite3
import json
import time
from pathlib import Path

RECORDING_DB = os.getenv("MUA_RECORDING_DB")

if RECORDING_DB is None:
    raise Exception("MUA_RECORDING_DB environment variable not set.")

RECORDING_DB = Path(RECORDING_DB)

cmd = sys.argv

if not cmd[0].endswith("-wrap"):
    raise Exception("This script should only be used with executables ending in '-wrap'.")

if cmd[0] == "/usr/bin/gclang-wrap":
    cmd[0] = "/root/go/bin/gclang"
elif cmd[0] == "/usr/bin/gclang++-wrap":
    cmd[0] = "/root/go/bin/gclang++"
else:
    raise Exception(f"Unknown executable: {cmd[0]}")

if not RECORDING_DB.exists():
    with sqlite3.connect(RECORDING_DB) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE cmds (time, cmd)")


with sqlite3.connect(RECORDING_DB) as conn:
    c = conn.cursor()
    c.execute("INSERT into cmds Values(?, ?)", (time.time(), json.dumps(cmd)))

os.execl(cmd[0], *cmd)
