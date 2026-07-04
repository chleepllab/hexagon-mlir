#!/usr/bin/env bash
#
# Usage: source env.sh
# Sets up all environment variables needed to run hexagon-mlir tests.
source mlir-env/bin/activate
HEXAGON_MLIR_ROOT="$(git rev-parse --show-toplevel)"
BASE_DIR="$(cd "$HEXAGON_MLIR_ROOT/.." && pwd)"
TRITON_ROOT="$HEXAGON_MLIR_ROOT/triton"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TRITON_BUILD="$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}"

export HEXAGON_MLIR_ROOT
export TRITON_ROOT
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export HEXAGON_ARCH_VERSION=75

export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export TRITON_SHARED_OPT_PATH="$TRITON_BUILD/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
export PYTHONPATH=$TRITON_ROOT/python:${PYTHONPATH:-}
export PATH=$TRITON_BUILD/third_party/qcom_hexagon_backend/bin:$TRITON_BUILD/third_party/triton_shared/tools/triton-shared-opt:$PATH

export HEXAGON_SDK_ROOT=${BASE_DIR}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/
export HEXAGON_TOOLS=${BASE_DIR}/HEXAGON_TOOLS/Tools
export HEXKL_ROOT=${BASE_DIR}/HEXKL_DIR/hexkl_addon
LINALG_OPT=$(which linalg-hexagon-opt)
export HEXAGON_RUNTIME_LIBS_DIR="$(dirname $LINALG_OPT)/runtime"

export HOST_TOOLCHAIN=${BASE_DIR}/HOST_TOOLCHAIN
export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"
export PATH="${HOST_TOOLCHAIN}/bin:$PATH"
ENV_DIR="${BASE_DIR}/mlir-env"
if [[ "$VIRTUAL_ENV" != "$ENV_DIR" ]]; then
    source "$ENV_DIR/bin/activate"
fi
export RUN_ON_SIM=1
#python test/python/torch-mlir/gpt2lmheadmodel.py
#  pytest -sv test/python/triton/test_flash_attention.py
#MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 pytest -sv matmul.py
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump_share pytest -sv base.py
#TRITON_ALWAYS_COMPILE=1 \
#  TRITON_KERNEL_DUMP=1 \
#  TRITON_DUMP_DIR=dump_share \
#  MLIR_ENABLE_DUMP=1 \
#  MLIR_DUMP_PATH=dump_share/mlir.log \
#  LLVM_IR_ENABLE_DUMP=1 \
#  2> dump_share/llir.log \
#  pytest -sv matmul.py
#pytest -sv matmul.py
