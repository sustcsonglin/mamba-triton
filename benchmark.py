
import time

import torch
from p_scan import pscan_selective_scan
from ref import ref_selective_scan
from selective_scan_interface import selective_scan_fn, selective_scan_ref
from triton_parallel_scan import triton_selective_scan
from triton_sequential_scan import triton_selective_scan_sequential

if __name__ == '__main__':
    B = 32
    T = 2048
    D = 1024
    K = 16
    dtype = torch.bfloat16
    A = (-(torch.rand(D, K, dtype=torch.float32)).exp().cuda()).requires_grad_(True)
    x = torch.randn(B, T, D, dtype=dtype).cuda().requires_grad_(True)
    delta = torch.randn(B, T, D, dtype=dtype).sigmoid().cuda().requires_grad_(True)
    B2 = torch.randn(B, T, K, dtype=dtype).cuda().requires_grad_(True)
    C = torch.randn(B, T, K, dtype=dtype).cuda().requires_grad_(True)
    D2 = torch.randn(D, dtype=torch.float32).cuda().requires_grad_(True)



    do = torch.randn_like(x)
    print("Warmup start")
    for _ in range(50):
        o, final_state = triton_selective_scan_sequential(x, delta, A, B2, C, D2)
        o.backward(do, retain_graph=True)
        o = selective_scan_fn(x, delta, A, B2, C, D2)
        o.backward(do, retain_graph=True)

    print("Warmup done")
    start = time.time()
    torch.cuda.synchronize()
    # with torch.no_grad():
    for _ in range(100):
        o, final_state = triton_selective_scan_sequential(x, delta, A, B2, C, D2)
        o.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    end = time.time()
    print('triton', end - start)

    start = time.time()
    torch.cuda.synchronize()
    # with torch.no_grad():
    for _ in range(100):
        o = selective_scan_fn(x, delta, A, B2, C, D2)
        o.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    end = time.time()
    print('cuda', end - start)

