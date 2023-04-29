import torch

A = torch.zeros(2, 64, 100)
B = torch.ones(2, 1024, 100)

C = torch.cat([A, B], dim=1)

print(f"A.shape: {A.shape}")
print(f"B.shape: {B.shape}")
print(f"C.shape: {C.shape}")
print(C[0])