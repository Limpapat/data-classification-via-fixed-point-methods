import torch

def constant_fn(x:torch.tensor, constant:float=0.):
    return torch.tensor(constant, dtype=torch.float)

def l1_fn(x:torch.tensor):
    return torch.norm(x, p=1)

def l2_fn(x:torch.tensor):
    return .5 * torch.norm(x, p=2)**2

if __name__ == '__main__':
    x = torch.tensor([1.,2.,3.], dtype=torch.float)
    print(constant_fn(x, 1.))
    print(l1_fn(x))
    print(l2_fn(x))
