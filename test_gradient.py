
import torch
from ref import ref_selective_scan
from triton_parallel_scan import triton_selective_scan
from triton_sequential_scan import triton_selective_scan_sequential

if __name__ == '__main__':
    B = 2
    T = 16
    D = 512
    K = 16
    dtype = torch.float32
    A = (-(torch.rand(D, K, dtype=dtype)).exp().cuda()).requires_grad_(True)
    x = torch.randn(B, T, D, dtype=dtype).cuda().requires_grad_(True)
    delta = torch.randn(B, T, D, dtype=dtype).sigmoid().cuda().requires_grad_(True)
    B2 = torch.randn(B, T, K, dtype=dtype).cuda().requires_grad_(True)
    C = torch.randn(B, T, K, dtype=dtype).cuda().requires_grad_(True)
    D2 = torch.randn(D, dtype=dtype).cuda().requires_grad_(True)

    initial_state = torch.randn(B, D, K, dtype=dtype).cuda().requires_grad_(False)

    tri, tri_final = triton_selective_scan_sequential(x, delta, A, B2, C, D2, initial_state)
    do = torch.randn_like(tri)
    tri.backward(do)

    tri_dc, C.grad = C.grad.clone(), None
    tri_dx, x.grad = x.grad.clone(), None
    tri_db, B2.grad = B2.grad.clone(), None
    tri_delta, delta.grad = delta.grad.clone(), None
    tri_A, A.grad = A.grad.clone(), None

    ref, ref_final = ref_selective_scan(x, delta, A, B2, C, D2, initial_state)

    print((tri-ref).abs().max())
    print((tri_final-ref_final).abs().max())

    ref.backward(do)
    ref_dc, C.grad = C.grad.clone(), None
    ref_dx, x.grad = x.grad.clone(), None
    ref_db, B2.grad = B2.grad.clone(), None
    ref_delta, delta.grad = delta.grad.clone(), None
    ref_A, A.grad = A.grad.clone(), None

    print((tri_dc-ref_dc).abs().max())    
    print((tri_dx-ref_dx).abs().max())
    print((tri_db-ref_db).abs().max())  
    print((tri_delta-ref_delta).abs().max())
    print((tri_A-ref_A).abs().max())
    breakpoint()


