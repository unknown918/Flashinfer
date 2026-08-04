import torch

num_layers = 32

for i in range(num_layers):
    a = torch.load(f"sparse_attn_output{i}.pt", map_location="cpu")
    b = torch.load(f"dense_attn_output{i}.pt", map_location="cpu")
    print(torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2))
