import torch
import triton
import triton.language as tl
from einops import einsum, rearrange, repeat
from numpy import dtype

assert triton.__version__ != '2.1.0', 'Triton 2.1.0 is missing enable_fp_fusion. Triton 2.2.0 is required for numerical stability of this implementation.'

inv_ln2 = 1.44269504

# credit: https://github.com/proger/accelerated-scan/blob/b9edbad65c673f9a1915efe51dc6bbf50fd7f8c4/accelerated_scan/triton.py

# manual tuple packing by @jackd from https://github.com/openai/triton/issues/2359
@triton.jit
def unpack64(merged):
    tl.static_assert(merged.dtype == tl.uint64)
    b = (merged & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    return a, b


@triton.jit
def pack64(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    a = a << 32
    b = b.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    return a | b


@triton.jit()
def first_order_op(l, r):
    """
    See https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf Section 1.4.1
    """
    xl, fl = unpack64(l)
    xr, fr = unpack64(r)
    x = xl * fr + xr
    f = fl * fr
    return pack64(x, f)


@triton.jit
def forward_scan(
    A, 
    B, 
    C, 
    Dt,
    X,
    Y, 
    H,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    
    dt_ptr = Dt + i_bh * T * D + i_v * T + tl.arange(0, T)
    dt = tl.load(dt_ptr).to(tl.float32)
    
    x_ptr = X + i_bh * T * D + i_v * T + tl.arange(0, T)
    x = tl.load(x_ptr).to(tl.float32)

    x_dt = x * dt

    y = tl.zeros([T,], dtype=tl.float32)


    for i in range(K):
        b_ptr = B + i_bh * T * K + i * T + tl.arange(0, T)
        c_ptr = C + i_bh * T * K + i * T + tl.arange(0, T)
        H_ptr = H + i_bh * T * D * K + i_v * K * T + i * T + tl.arange(0, T) 
        b = tl.load(b_ptr).to(tl.float32)
        c = tl.load(c_ptr).to(tl.float32)
        x_dt_b = x_dt * b
        a = tl.load(A + i_v * K + i).to(tl.float32)
        dt_a = tl.exp(dt * a)
        tuples = pack64(x_dt_b, dt_a)
        output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=first_order_op)
        o, _ = unpack64(output_tuples_)
        tl.store(H_ptr, o)
        y += (o * c)
        
    y_ptr = Y + i_bh * T * D + i_v * T + tl.arange(0, T)
    tl.store(y_ptr, y)


@triton.jit
def backward_scan_du_delta_A(
    A,
    B,
    C,
    U,
    Dt,
    DO,
    H,
    DA, 
    DB,
    DC,
    dDt,
    dU,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)

    dt_ptr = Dt + i_bh * T * D + i_v * T + T - tl.arange(0, T)
    dt = tl.load(dt_ptr, mask=tl.arange(0, T) > 0, other=0) 

    dt_ptr2 = Dt + i_bh * T * D + i_v * T + T - 1 - tl.arange(0, T)
    dt2 = tl.load(dt_ptr2)

    dO_ptr = DO + i_bh * T * D + i_v * T + T-1 - tl.arange(0, T)
    do = tl.load(dO_ptr)

    d_delta = tl.zeros([T], dtype=tl.float32)
    d_u = tl.zeros([T], dtype=tl.float32)

    u_ptr = U + i_bh * T * D + i_v * T + T-1 - tl.arange(0, T)
    u = tl.load(u_ptr).to(tl.float32)

    for i in range(K):
        H_ptr2 = H + i_bh * T * D * K + i_v * K * T + i * T + (T-1) - tl.arange(0, T)
        h2 = tl.load(H_ptr2)
        dc = h2 * do
        dc_ptr = DC + i_bh * T * K * D + i * D * T + T - 1 - tl.arange(0, T) + i_v * T
        tl.store(dc_ptr, dc)
                 
        c_ptr = C + i_bh * T * K + i * T + T - 1 - tl.arange(0, T)
        b_ptr = B + i_bh * T * K + i * T + T - 1 - tl.arange(0, T)
        # DH_ptr = DH + i_bh * T * D * K + i_v * K * T + i * T + T - 1 - tl.arange(0, T)
        b = tl.load(b_ptr).to(tl.float32)
        c = tl.load(c_ptr).to(tl.float32)
        a = tl.load(A + i_v * K + i).to(tl.float32)
        dt_a = tl.math.exp(dt * a)
        dt_a = tl.where(tl.arange(0, T) > 0, dt_a, 0)
        do_c = c * do
        tuples = pack64(do_c, dt_a)
        output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=first_order_op)
        dh, _ = unpack64(output_tuples_)

        # gradient wrt input u
        d_u += dh * dt2 * b
        d_delta += dh * b * u
        d_b = dh * u * dt2

        db_ptr = DB + i_bh * T * K * D + i * D * T + T - 1 - tl.arange(0, T) + i_v * T
        tl.store(db_ptr, d_b)


        # gradient wrt decay
        H_ptr = H + i_bh * T * D * K + i_v * K * T + i * T + (T-2) - tl.arange(0, T)
        h = tl.load(H_ptr, mask=tl.arange(0, T) < T-1, other=0)
        d_decay = h * dh * tl.exp(dt2 * a)
        d_delta += d_decay * a
        d_a = tl.sum(d_decay * dt2)
        tl.store(DA + i_bh * K * D + i_v * K + i, d_a)
    
    tl.store(dU + i_bh * T * D + i_v * T + T-1 - tl.arange(0, T), d_u)
    tl.store(dDt + i_bh * T * D + i_v * T + T-1 - tl.arange(0, T), d_delta)
    
# @triton.jit
# def grad_compute(
#     A, B, C, Dt, U, H, 
#     dA, dB, dC, dDt, 
#     dO, dH, dU, 
#     batch, 
#     T: tl.constexpr,
#     DV: tl.constexpr,
#     DK: tl.constexpr,
#     BV: tl.constexpr,
# ):

#     i_bh = tl.program_id(0)
#     i_v = tl.program_id(1)

#     prev_h = tl.zeros([BV, DK], dtype=tl.float32)
#     dA_acc = tl.zeros([BV, DK], dtype=tl.float32)

#     # [BV, DK]
#     A = tl.load(A + (tl.arange(0, BV)[:, None] + i_v * BV) * DK + tl.arange(0, DK)[None, :])

#     H_ptr = H + i_bh * T * DK * DV + (i_v * BV + tl.arange(0, BV)[:, None]) * DK + tl.arange(0, DK)[None, :]
#     dH_ptr = dH + i_bh * T * DK * DV + (i_v * BV + tl.arange(0, BV)[:, None]) * DK + tl.arange(0, DK)[None, :]

#     C_ptr = C + i_bh * T * DK + tl.arange(0, DK)
#     dC_ptr = dC + (i_bh + i_v * batch) * T * DK + tl.arange(0, DK)

#     B_ptr = B + i_bh * T * DK + tl.arange(0, DK)
#     dB_ptr = dB + (i_bh + i_v * batch) * T * DK + tl.arange(0, DK)

#     Dt_ptr = Dt + i_bh * T * DV + i_v * BV + tl.arange(0, BV)
#     dDt_ptr = dDt + i_bh * T * DV + i_v * BV + tl.arange(0, BV)

#     u_ptr = U + i_bh * T * DV + i_v * BV + tl.arange(0, BV)
#     du_ptr = dU + i_bh * T * DV + i_v * BV + tl.arange(0, BV)
#     do_ptr = dO + i_bh * T * DV + i_v * BV + tl.arange(0, BV)
    
#     for i in range(T):
#         h = tl.load(H_ptr)
#         dh = tl.load(dH_ptr)
#         b = tl.load(B_ptr)
#         delta = tl.load(Dt_ptr).to(tl.float32)
#         u = tl.load(u_ptr)
#         do = tl.load(do_ptr)

#         # gradient wrt output proj
#         dc = tl.sum(do[:, None] * h, axis=0)
#         tl.store(dC_ptr, dc)

#         # gradient wrt input
#         db = tl.sum(dh * u[:, None] * delta[:, None], axis=0)
#         du_delta = tl.sum(dh * b[None, :], axis=1)
#         d_delta = du_delta * u
#         du = du_delta * delta
#         tl.store(dB_ptr, db)
#         tl.store(du_ptr, du)

#         # gradient wrt decay
#         d_decay = prev_h * dh
#         gate = tl.exp(delta[:, None] * A)
#         d_decay *= gate
#         dA_acc += d_decay * delta[:, None]
#         d_delta += tl.sum(d_decay * A, axis=1)
#         prev_h = h

#         tl.store(dDt_ptr, d_delta.to(dDt.dtype.element_ty))

#         # update ptrs
#         H_ptr += DK * DV
#         dH_ptr += DK * DV
        
#         B_ptr += DK
#         dB_ptr += DK

#         dDt_ptr += DV
#         Dt_ptr += DV
        
#         u_ptr += DV
#         du_ptr += DV

#         do_ptr += DV
#         dC_ptr += DK
    
#     #fp32
#     dA_ptr = dA + i_bh * DV * DK + (tl.arange(0, BV)[:, None] + i_v * BV) * DK + tl.arange(0, DK)[None, :]
#     tl.store(dA_ptr, dA_acc)

class SelectiveScan(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, u, delta, A, B, C):
        b_size,  T, d = u.shape
        K = B.shape[-1]

        ctx.b_size = b_size 
        ctx.T = T
        ctx.d = d
        ctx.K = K

        u = u.transpose(-1, -2).contiguous()
        delta = delta.transpose(-1, -2).contiguous()
        B = B.transpose(-1, -2).contiguous()
        C = C.transpose(-1, -2).contiguous()
        o = torch.empty_like(u)
        H = torch.empty(b_size, d, K, T, device=u.device, dtype=torch.float32)
        forward_scan[(b_size, d)](A, B, C, delta, u, o, H, T, d, K)
        ctx.save_for_backward(A, B, C, delta, H, u)
        return o.transpose(-1, -2).contiguous()

    
    @staticmethod
    def backward(ctx, grad_output):
        do = grad_output.transpose(-1, -2).contiguous()
        A, B, C, delta, H, u = ctx.saved_tensors
        b_size = ctx.b_size
        T = ctx.T
        d = ctx.d
        K = ctx.K
        
        dA = A.new_empty(b_size, d, K)
        du = torch.empty_like(u)
        d_delta = torch.empty_like(delta)
        db = B.new_empty(b_size, K, d, T)
        dc = C.new_empty(b_size, K, d, T)

        backward_scan_du_delta_A[(b_size, d)](A, B, C, u, delta, do, H, dA, db, dc, d_delta, du, T, d, K)
        db = db.sum(-2)
        dc = dc.sum(-2)

        return du.transpose(-1, -2), d_delta.transpose(-1, -2), dA.sum(0), db.transpose(-1, -2), dc.transpose(-1, -2)

        
def triton_selective_scan(u, delta, A, B, C, D):
    original_dtype = u.dtype
    D = D.float()
    A = A.float()
    o = SelectiveScan.apply(u, delta, A, B, C)
    o += D * u 
    return o.to(original_dtype)


