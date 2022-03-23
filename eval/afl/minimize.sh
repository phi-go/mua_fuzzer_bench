#!/usr/bin/env bash

_term() {
  echo "Caught SIGINT signal!"
  kill -INT "$child"
}

trap _term SIGINT

set -Euxo pipefail

echo "workdir: $(pwd)"

export LD_LIBRARY_PATH=/home/user/lib/

afl-clang-fast++ -o put -v /home/user/lib/libdynamiclibrary.so /home/user/common/main.cc $1 $2

shift
shift

SEEDS_IN="$1"
SEEDS_OUT="$2"

shift
shift

export AFL_NO_AFFINITY=1

echo "afl-cmin -i "$SEEDS_IN" -o out_seeds -- ./put $@ @@"
afl-cmin -i "$SEEDS_IN" -o out_seeds -- ./put $@ @@

mv out_seeds/* "$SEEDS_OUT"/

