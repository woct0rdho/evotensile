#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

ROCM_PATH=${ROCM_PATH:-$HOME/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel}
HIPBLASLT_ROOT=${HIPBLASLT_ROOT:-$HOME/rocm-libraries/projects/hipblaslt}
TENSILELITE_BUILD=${TENSILELITE_BUILD:-$HOME/rocm-libraries/build/tensilelite-client}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build}
GFX_ARCH=${GFX_ARCH:-gfx1151}
CXX=${CXX:-$ROCM_PATH/lib/llvm/bin/amdclang++}

OUT=$BUILD_DIR/evotensile-structured-runner
OBJ=$BUILD_DIR/structured_runner.o
mkdir -p "$BUILD_DIR"

OPENBLAS_CFLAGS=()
OPENBLAS_LIBS=()
if [[ ${EVOTENSILE_USE_OPENBLAS:-1} != 0 ]]; then
  if pkg-config --exists openblas; then
    # shellcheck disable=SC2207
    OPENBLAS_CFLAGS=($(pkg-config --cflags openblas))
    # shellcheck disable=SC2207
    OPENBLAS_LIBS=($(pkg-config --libs openblas))
    OPENBLAS_CFLAGS+=(-DEVOTENSILE_USE_OPENBLAS)
  else
    echo "OpenBLAS pkg-config metadata not found; building structured runner with scalar reference fallback" >&2
  fi
fi

COMMON_A=$TENSILELITE_BUILD/tensilelite/client/libtensilelite-client-common.a
ORIGAMI_A=$TENSILELITE_BUILD/origami/liborigami.a
MXDATAGEN_A=$TENSILELITE_BUILD/clients/common/libhipblaslt-mxdatagen.a

for path in "$CXX" "$COMMON_A" "$ORIGAMI_A" "$MXDATAGEN_A"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required build artifact: $path" >&2
    echo "build TensileLite client first, e.g. ~/rocm-libraries/build_tensilelite_client.sh" >&2
    exit 1
  fi
done

COMMON_FLAGS=(
  -std=gnu++20 -O3 -DNDEBUG
  -DHIPBLASLT_ENABLE_MXDATAGENERATOR
  -DORIGAMI_STATIC
  -DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK=0
  -DTENSILE_DEFAULT_SERIALIZATION
  -DTENSILE_USE_HIP
  -DTENSILE_YAML
  -DUSE_PROF_API=1
  -D__HIP_HCC_COMPAT_MODE__=1
  -D__HIP_PLATFORM_AMD__=1
  -I"$HIPBLASLT_ROOT/tensilelite/client/include"
  -I"$HIPBLASLT_ROOT/tensilelite/include"
  -I"$HIPBLASLT_ROOT/tensilelite/rocisa"
  -I"$HIPBLASLT_ROOT/../../shared/origami/include"
  -I"$TENSILELITE_BUILD/origami/include"
  -I"$HIPBLASLT_ROOT/clients/common/include"
  -I"$HIPBLASLT_ROOT/library/include"
  -I"$TENSILELITE_BUILD/library/include"
  -I"$HIPBLASLT_ROOT/../../shared/mxdatagenerator/lib/include"
  -isystem "$ROCM_PATH/include"
  "${OPENBLAS_CFLAGS[@]}"
)

"$CXX" "${COMMON_FLAGS[@]}" -x hip --offload-arch="$GFX_ARCH" \
  -c "$REPO_ROOT/csrc/structured_runner.cpp" -o "$OBJ"

"$CXX" -O3 -DNDEBUG "$OBJ" \
  -o "$OUT" \
  -L"$ROCM_PATH/lib" \
  -L"$ROCM_PATH/lib/llvm/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib" \
  "$COMMON_A" \
  -lLLVM-23git \
  "$ORIGAMI_A" \
  -lamd_smi \
  "$MXDATAGEN_A" \
  --hip-link --offload-arch="$GFX_ARCH" \
  -lamdhip64 -lomp -ldl -lpthread -lm -lrt -lz -lzstd \
  "${OPENBLAS_LIBS[@]}"

echo "$OUT"
