import torch
import torch.nn as nn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from einops import rearrange, repeat

model_dtype = torch.bfloat16 # follows model dtype
chunk_size = 64
B, T, H, D = 2, 128, 8, 64  # batch, seq, heads, head_dim

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        normed = self.weight * hidden_states.to(input_dtype)
        return normed.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
q_norm = Qwen3RMSNorm(hidden_size=D).to("cuda:0")
k_norm = Qwen3RMSNorm(hidden_size=D).to("cuda:0")

def l2norm_fwd_torch(x: torch.Tensor, eps: float = 1e-6): #, output_dtype: torch.dtype ):
    """
    Row-wise L2 normalization (per last dim), mirroring the Triton kernel:
        rstd = 1 / sqrt(sum(x^2) + eps)
        y    = x * rstd
    Returns:
        y: same shape as x (dtype = output_dtype if given, else x.dtype)
        rstd: shape x.shape[:-1] (float32)
    """
    # breakpoint()
    dtype_orig = x.dtype

    x_shape = x.shape
    D = x_shape[-1]

    # Flatten to rows of length D
    x2d = x.reshape(-1, D)

    # Kernel does math in fp32
    x_fp32 = x2d.to(torch.float32)

    # rstd per row
    rstd = torch.rsqrt(torch.sum(x_fp32 * x_fp32, dim=-1) + eps)  # (T,)

    # Normalize and cast back
    y2d = x_fp32 * rstd.unsqueeze(-1)  # (T, D)
    y = y2d.to(x.dtype).reshape(x_shape)

    # rstd is float32, shaped to drop last dim
    rstd = rstd.reshape(x_shape[:-1])

    return y.to(dtype_orig), rstd

def transform_ref(q, k, v, g, beta, use_qk_l2norm_in_kernel):
    ### added to align with naive codes ###
    q_T = rearrange(q, "b l h d -> b h l d")
    if use_qk_l2norm_in_kernel:
        q_T, _ = l2norm_fwd_torch(q_T)
    k_T = rearrange(k, "b l h d -> b h l d")
    if use_qk_l2norm_in_kernel:
        k_T, _ = l2norm_fwd_torch(q_T)
    v_T = rearrange(v, "b l h d -> b h l d")
    g_T = rearrange(g, "b l h -> b h l")
    beta_T = rearrange(beta, "b l h -> b h l")
    #####################################
    return q_T, k_T, v_T, g_T, beta_T


def pad_chunk_dimNeg1(chunk_sz, tensors):
    return [F.pad(x, (0, chunk_sz - x.size(-1) % chunk_sz), "constant", 0) if x is not None else None for x in tensors]

def pad_chunk_dimNeg2(chunk_sz, tensors):
    return [F.pad(x, (0,0,0, chunk_sz - x.size(-2) % chunk_sz), "constant", 0) if x is not None else None for x in tensors]

def unpad_chunk_dim1(l, tensors):
    return [x[:,:l] if x is not None else None for x in tensors]

def unpad_chunk_dim2(l, tensors):
    return [x[:,:,:l] if x is not None else None for x in tensors]

def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int, #chunk size
    initial_state: torch.Tensor=None,
    use_qk_l2norm_in_kernel: bool=True
):
    """Adapted from: https://github.com/NVlabs/GatedDeltaNet/blob/4decc9b84e7e27b6400fea562096fdd35fe7ef2e/lit_gpt/gated_delta_rule_ops/chunk.py#L639
    
    An extra argument `initial_state` is passed for `S`
    """
    # breakpoint()
    
    dtype_ = torch.float32
    q, k, v, g, beta = transform_ref(q, k, v, g, beta, use_qk_l2norm_in_kernel) # added for adaptation
    b, h, l, d_k = q.shape
    l_padded = l
    chunk_size = BT
    # chunk_size = min(BT, l) # added for adaptation; using when there is no padding

    # print ("preI")
    # [print(x.shape) if x is not None else None for x in (q, k, v, g, beta, initial_state)]
    if l % chunk_size != 0:
        q, k, v = pad_chunk_dimNeg2(chunk_size, (q,k,v)) # added as tmp fix
        beta, g = pad_chunk_dimNeg1(chunk_size, (beta,g)) # added as tmp fix
    l_padded = q.shape[2]
    # print ("postI")
    # [print(x.shape) if x is not None else None for x in (q, k, v, g, beta, initial_state)]
    # alias
    q, k, v, beta, g = map(lambda x: x.to(dtype_), [q, k, v, beta, g])
    decay = g
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # assert l % chunk_size == 0, (l, chunk_size)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c = chunk_size), [q, k, v, k_beta, decay.unsqueeze(-1)])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).exp()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=dtype_, device=q.device)
    attn = attn
    k_cumsum = attn @ v
    attn = -((k_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=dtype_, device=q.device)
    attn = attn
    w = k_cumdecay = attn @ k_beta
    u = v = k_cumsum
    # S = k.new_zeros(b, h, d_k, d_v)
    S = k.new_zeros(b, h, d_k, d_v) if initial_state is None else initial_state
    # breakpoint()
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l_padded // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new

    # w = rearrange(w, 'b h n c d -> b h (n c) d')
    # u = rearrange(u, 'b h n c d -> b h (n c) d')
    o = rearrange(o, 'b h n c d -> b (n c) h d') # added for adaptation
    # print ("preO")
    # [print(x.shape) if x is not None else None for x in (o, w, u, S)]
    if l % chunk_size != 0:
        o, = unpad_chunk_dim1(l, (o,)) # added as tmp fix
        # w, u = unpad_chunk_dim2(l, (w, u)) # added as tmp fix
    # breakpoint()

    return o, S



def has_nan(x):
    return torch.isnan(x).any().item()


def report_nan_positions(name, x, max_print: int = 10):
    mask = torch.isnan(x)
    count = int(mask.sum().item())
    if count == 0:
        print(f"{name}: no NaNs")
        return
    print(f"{name}: NaNs={count}")
    idx = mask.nonzero(as_tuple=False)
    if idx.numel() > 0:
        # Show a few example indices to locate the issue quickly.
        print(f"  sample positions (up to {max_print}): {idx[:max_print].cpu().tolist()}")


def run_one_test(use_rms_norm: bool, use_qk_l2norm_in_kernel: bool, device: str = "cuda"):
    torch.manual_seed(0)

    dtype = torch.bfloat16       # use the same dtype you use in training, e.g. bf16/fp16

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)

    # g has shape [B, T, H] in your GatedDeltaNet / TMix_qwen3gdn usage
    g = torch.randn(B, T, H, dtype=torch.float32, device=device)

    # beta is [B, T, H] as well (output of a proj + sigmoid)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device)

    # no initial state (same as TMix when last_state is None)
    initial_state = None

    # simple contiguous cu_seqlens: [0, T, 2T, ...]
    # cu_seqlens = torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.int32)
    cu_seqlens = None

    if use_rms_norm:
        q = q_norm(q)
        k = k_norm(k)

    print(f"use_qk_l2norm_in_kernel={use_qk_l2norm_in_kernel}")
    o, recurrent_state = chunk_gated_delta_rule_ref(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        BT=chunk_size,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    print ('====================================ref========================')
    print("  o has NaN:", has_nan(o), o.shape)
    report_nan_positions("o", o)
    print("  inputs:")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        report_nan_positions(name, tensor)
    if recurrent_state is not None:
        print("  recurrent_state has NaN:", has_nan(recurrent_state))
        report_nan_positions("recurrent_state", recurrent_state)
    print ()
        
    print ('====================================triton========================')
    o, recurrent_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    breakpoint()
    print("  o has NaN:", has_nan(o), o.shape)
    report_nan_positions("o", o)
    print("  inputs:")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        report_nan_positions(name, tensor)
    if recurrent_state is not None:
        print("  recurrent_state has NaN:", has_nan(recurrent_state))
        report_nan_positions("recurrent_state", recurrent_state)
    print ()

    print ()


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Reference case (what GatedDeltaNet uses in this repo)
    # run_one_test(use_rms_norm=True, use_qk_l2norm_in_kernel=True, device=device)

    # Suspected-bug case
    run_one_test(use_rms_norm=True, use_qk_l2norm_in_kernel=True, device=device)