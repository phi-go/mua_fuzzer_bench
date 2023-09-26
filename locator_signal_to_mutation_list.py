
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-signal-dir", required=True,
        help='The trigger_signal dir.')
    parser.add_argument("--prog", required=True,
        help='The trigger_signal dir.')
    parser.add_argument("--out", required=True,
        help='The path of the output json file.')

    args = parser.parse_args()

    ts_dir = Path(args.trigger_signal_dir)
    assert ts_dir.is_dir()

    signal_paths = list(ts_dir.glob("*"))
    triggered = []
    for sp in signal_paths:
        assert sp.is_file()
        triggered.append(int(sp.name))

    with open(args.out, "w") as f:
        json.dump([{
            "prog": args.prog,
            "mutation_ids": triggered,
            "mode": "single",
        }], f)



if __name__ == "__main__":
    main()