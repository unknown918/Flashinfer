# q, k = flashinfer.apply_llama31_rope(
#     q,
#     k,
#     torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device),
#     torch.tensor([0], dtype=torch.int32, device=q.device),
# )