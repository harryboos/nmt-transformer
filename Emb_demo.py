import torch
emb = torch.rand((10, 7))
print(emb)

batch = torch.Tensor([[1, 2, 3],[5, 3, 1]]).long()
print(batch)
print(emb[batch])
print(emb[batch].shape)