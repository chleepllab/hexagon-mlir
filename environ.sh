#!/usr/bin/env bash
#
# Usage: source env.sh
# Sets up all environment variables needed to run hexagon-mlir tests.
#
 
HEXAGON_MLIR_ROOT="$(git rev-parse --show-toplevel)"
BASE_DIR="$(cd "$HEXAGON_MLIR_ROOT/.." && pwd)"
TRITON_ROOT="$HEXAGON_MLIR_ROOT/triton"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TRITON_BUILD="$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}"
 
export HEXAGON_MLIR_ROOT
export TRITON_ROOT
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export HEXAGON_ARCH_VERSION=75
 
# Triton plugin discovery
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export TRITON_SHARED_OPT_PATH="$TRITON_BUILD/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
export PYTHONPATH=$TRITON_ROOT/python:${PYTHONPATH:-}
 
# PATH must be set before any 'which' calls below
export PATH=$TRITON_BUILD/third_party/qcom_hexagon_backend/bin:$TRITON_BUILD/third_party/triton_shared/tools/triton-shared-opt:$PATH
 
# Hexagon SDK / Tools
export HEXAGON_SDK_ROOT=${BASE_DIR}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/
export HEXAGON_TOOLS=${BASE_DIR}/HEXAGON_TOOLS/Tools
export HEXKL_ROOT=${BASE_DIR}/HEXKL_DIR/hexkl_addon
 
# Runtime libs dir — same bin/ directory as linalg-hexagon-opt, under runtime/
export HEXAGON_RUNTIME_LIBS_DIR="$TRITON_BUILD/third_party/qcom_hexagon_backend/bin/runtime"
 
# Host toolchain
export HOST_TOOLCHAIN=${BASE_DIR}/HOST_TOOLCHAIN
export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"
export PATH="${HOST_TOOLCHAIN}/bin:$PATH"
 
# Use hexagon-sim instead of Android device
export RUN_ON_SIM=1
 
# Activate venv if not already active
ENV_DIR="${BASE_DIR}/mlir-env"
if [[ "$VIRTUAL_ENV" != "$ENV_DIR" ]]; then
    source "$ENV_DIR/bin/activate"
fi
 
echo "hexagon-mlir env ready (Python ${PYTHON_VERSION})"
echo "linalg-hexagon-opt: $(which linalg-hexagon-opt)"
echo "HEXAGON_RUNTIME_LIBS_DIR: $HEXAGON_RUNTIME_LIBS_DIR"
