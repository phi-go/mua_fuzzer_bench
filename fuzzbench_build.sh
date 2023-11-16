#!/usr/bin/env bash

set -Eeuox pipefail

# touch /out/testentry
/bin/mua_build_benchmark
cd /mutator && gradle build
ldconfig /mutator/build/install/LLVM_Mutation_Tool/lib/
pipx run hatch run src/mua_fuzzer_benchmark/eval.py locator_local --config-path /tmp/config.json --result-path /tmp/test/
# cd /tmp && /tmp/test/progs/'+fuzz_target+'/'+fuzz_target+'.locator /benchmark.yaml
# cd /mutator && python locator_signal_to_mutation_list.py --trigger-signal-dir /tmp/trigger_signal/ --prog xml --out /out/mua_all_list.json
cp /tmp/test/progs/**/*.locator /out/
cp /tmp/config.json /out/config.json
