import pytest
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())
ATOL = 5.0

# Kernel 1: A * B = C
@triton.jit
def matmul_kernel_1(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    A_block_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(K, 1), offsets=(0, 0), block_shape=(M, K), order=(1, 0))
    B_block_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(N, 1), offsets=(0, 0), block_shape=(K, N), order=(1, 0))
    C_block_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(N, 1), offsets=(0, 0), block_shape=(M, N), order=(1, 0))

    q = tl.load(A_block_ptr)
    k_t = tl.load(B_block_ptr)
    qk = tl.dot(q, k_t, out_dtype=C.type.element_ty)
    tl.store(C_block_ptr, qk)
    # Removed `return qk` because top-level kernels must not return values

# Kernel 2: C * D = E
@triton.jit
def matmul_kernel_2(C, D, E, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # Added block pointer & load for C, since it's passed as a base pointer
    C_block_ptr = tl.make_block_ptr(base=C, shape=(M, K), strides=(K, 1), offsets=(0, 0), block_shape=(M, K), order=(1, 0))
    D_block_ptr = tl.make_block_ptr(base=D, shape=(K, N), strides=(N, 1), offsets=(0, 0), block_shape=(K, N), order=(1, 0))
    E_block_ptr = tl.make_block_ptr(base=E, shape=(M, N), strides=(N, 1), offsets=(0, 0), block_shape=(M, N), order=(1, 0))

    c_tensor = tl.load(C_block_ptr)
    d = tl.load(D_block_ptr)
    res = tl.dot(c_tensor, d, out_dtype=E.type.element_ty)
    tl.store(E_block_ptr, res)


class TwoMatmulsModule(torch.nn.Module):
    def forward(self, mat_A, mat_B, mat_C, mat_D, mat_E, M, N1, K1, N2, K2):
        matmul_kernel_1[(1,)](mat_A, mat_B, mat_C, M, N1, K1,
            enableMultiThreading=True, enableVTCMTiling=True,
            enableConvertToHexagonmem=True, enableHexagonmemCopyToDMA=True,
        )
        matmul_kernel_2[(1,)](mat_C, mat_D, mat_E, M, N2, K2,
            enableMultiThreading=True, enableVTCMTiling=True,
            enableConvertToHexagonmem=True, enableHexagonmemCopyToDMA=True,
        )
        return mat_E


def hexagon_backend(gm: torch.fx.GraphModule, example_inputs):
    return gm.forward


@pytest.mark.parametrize("num_rows", [256])
@pytest.mark.parametrize("num_cols", [128])
@pytest.mark.parametrize("num_inner", [16])
def test_matmul_two_calls(num_rows, num_cols, num_inner):
    mat_A = torch.rand((num_rows, num_inner), dtype=torch.float16)
    mat_B = torch.rand((num_inner, num_cols), dtype=torch.float16)
    mat_C = torch.zeros((num_rows, num_cols), dtype=torch.float16)
    mat_D = torch.rand((num_cols, num_cols), dtype=torch.float16)
    mat_E = torch.zeros((num_rows, num_cols), dtype=torch.float16)

    model = TwoMatmulsModule()
    compiled_model = torch.compile(model, backend=hexagon_backend)
    
    # Swapped `mat_D` and `mat_C` back to the correct order required by `forward()`
    compiled_model(mat_A, mat_B, mat_C, mat_D, mat_E, num_rows, num_cols, num_inner, num_cols, num_cols)

    reference_C = torch.matmul(mat_A, mat_B)
    reference_E = torch.matmul(reference_C, mat_D)
    assert torch.allclose(mat_E, reference_E, atol=ATOL)
