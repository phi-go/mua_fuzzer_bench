#!/usr/bin/env bash

set -Eeuox pipefail

# touch /out/testentry
/bin/mua_build_benchmark
cd /mutator && gradle build
ldconfig /mutator/build/install/LLVM_Mutation_Tool/lib/
pipx run hatch run src/mua_fuzzer_benchmark/eval.py locator_local --config-path /mua_build/config.json --result-path /mua_build/
cp /mua_build/progs/**/*.locator $OUT/
cp /mua_build/config.json /out/config.json
