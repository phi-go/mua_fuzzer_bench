#!/usr/bin/env python3

import os, time

run_before_file = "/firstrun"

print("Entering mua_idle.py")

if(os.path.isfile(run_before_file)):
    while(True):
        print("IDLE IDLE IDLE")
        time.sleep(1000)
fp = open(run_before_file, 'x')
fp.close()