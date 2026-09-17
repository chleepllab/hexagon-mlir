# ===- test_matmul.py -------------------------------------------------------===
import pytest
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())
ATOL = 5.0
# Relative tolerance: both tl.dot calls accumulate in fp16, so absolute error
# grows with reduction depth (K) while relative error stays roughly flat.
# Measured max relative error ~0.5-0.6% at K=32/64 vs a fp32-exact reference;
# 2% leaves headroom for larger K without needing atol re-tuned each time.
RTOL = 0.02

# Kernel 1: A * B = C
@triton.jit
def matmul_kernel_1(A, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    A_block_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(K, 1), offsets=(0, 0), block_shape=(M, K), order=(1, 0))
    B_block_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(N, 1), offsets=(0, 0), block_shape=(K, N), order=(1, 0))
    C_block_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(N, 1), offsets=(0, 0), block_shape=(M, N), order=(1, 0))

    q = tl.load(A_block_ptr)
    k_t = tl.load(B_block_ptr)
    qk = tl.dot(q, k_t, out_dtype=C.type.element_ty)
    tl.store(C_block_ptr, qk)
    return qk

# Kernel 2: C * D = E
@triton.jit
def matmul_kernel_2(C_tensor, D, E, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    D_block_ptr = tl.make_block_ptr(base=D, shape=(K, N), strides=(N, 1), offsets=(0, 0), block_shape=(K, N), order=(1, 0))
    E_block_ptr = tl.make_block_ptr(base=E, shape=(M, N), strides=(N, 1), offsets=(0, 0), block_shape=(M, N), order=(1, 0))

    d = tl.load(D_block_ptr)
    res = tl.dot(C_tensor, d, out_dtype=E.type.element_ty)
    tl.store(E_block_ptr, res)

# Wrapper to submit both kernels to the compiler as one unit
@triton.jit
def fused_launcher(A, B, C, D, E, M: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, N: tl.constexpr):
    c_tensor = matmul_kernel_1(A, B, C, M, K1, K2)
    matmul_kernel_2(c_tensor, D, E, M, K2, N)

class TwoMatmulsModule(torch.nn.Module):
    def forward(self, mat_A, mat_B, mat_D, mat_C, mat_E, M, K1, K2, N):
        fused_launcher[(1,)](
            mat_A, mat_B, mat_C, mat_D, mat_E, M, K1, K2, N,
            enableMultiThreading=True, enableVTCMTiling=True,
            enableConvertToHexagonmem=True, enableHexagonmemCopyToDMA=True,
        )
        return mat_E

def hexagon_backend(gm: torch.fx.GraphModule, example_inputs):
    return gm.forward

@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("K1", [128])
@pytest.mark.parametrize("K2", [128])
@pytest.mark.parametrize("N", [128])
def test_matmul_two_calls(M, K1, K2, N):
    mat_A = torch.rand((M, K1), dtype=torch.float16)
    mat_B = torch.rand((K1, K2), dtype=torch.float16)
    mat_C = torch.zeros((M, K2), dtype=torch.float16)
    mat_D = torch.rand((K2, N), dtype=torch.float16)
    mat_E = torch.zeros((M, N), dtype=torch.float16)

    model = TwoMatmulsModule()
    compiled_model = torch.compile(model, backend=hexagon_backend)
    compiled_model(mat_A, mat_B, mat_D, mat_C, mat_E, M, K1, K2, N)

    reference_C = torch.matmul(mat_A, mat_B)
    reference_E = torch.matmul(reference_C, mat_D)
    # Both tl.dot calls accumulate in fp16 (out_dtype=fp16), so the absolute
    # error grows with reduction depth (K) - measured ~0.5-0.6% relative error
    # at K=32/64 vs a fp32-exact reference, dominated by fp16 accumulation on
    # Hexagon rather than by tiling/fusion. Use a relative tolerance so it
    # tracks K instead of needing re-tuning every time K changes.
    assert torch.allclose(mat_E, reference_E, atol=ATOL, rtol=RTOL)
