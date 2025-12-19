from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
import torch
from torch.optim import SGD


key_dim = 4
conv_size = 2
conv_bias = False
conv_custom_init = True

print ()

q_conv1d = ShortConvolution(
    hidden_size=key_dim,
    kernel_size=conv_size,
    bias=conv_bias,
    activation='silu',
).to("cuda:0")

print (f"{q_conv1d=}\n")
print (f"{q_conv1d.weight=}")
if conv_custom_init:
    q_conv1d.weight.data.copy_(
        torch.arange(
            q_conv1d.weight.numel(),
            device=q_conv1d.weight.device,
            dtype=q_conv1d.weight.dtype
        ).reshape_as(q_conv1d.weight)
    )
print (f"{q_conv1d.weight=} after zero_\n")

conv_state_q, conv_state_k, conv_state_v = None, None, None
use_cache = True
cu_seqlens = None

B, T, D = 1, 6, 2

q = torch.ones((B, T, D)).to("cuda:0")
q, conv_state_q = q_conv1d(
    x=q,
    cache=conv_state_q,
    output_final_state=use_cache,
    cu_seqlens=cu_seqlens,
)

optimizer = SGD(q_conv1d.parameters(), lr=0.1)

print (f"{q=}")

loss = torch.linalg.vector_norm(torch.ones_like(q).to("cuda:0") - q, dim = 0).mean()
loss.backward()


print (f"{loss=}")
print (f"{q.grad=}")
print (f"{q_conv1d.weight.grad=}")
print (f"{conv_state_q=}")

print ("step!")

optimizer.step()

print (f"{q_conv1d.weight=}")
print (f"{conv_state_q=}")