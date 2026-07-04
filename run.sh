#!/usr/bin/env bash
#
# Usage: source run.sh
# Sets up all environment variables needed to run hexagon-mlir tests.
source mlir-env/bin/activate
bash ./scripts/build_hexagon_mlir.sh
./env.sh
