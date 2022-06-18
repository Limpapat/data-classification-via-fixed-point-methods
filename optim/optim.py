from torch.optim.optimizer import Optimizer, required
from math import sqrt
import torch

"""
    Name : Forward Backward Algorithm (FBA) : A base calss for forward backward splitting type algorithms
        Parameters :
            lr : learning rate : a positive real number
            lam : regularization parameter : a positive real number
        Math Equation:
            x = PROX_lr*lam*G(x - lr*grad(x))
            where PROX_f is a proximity operator of f
"""
class FBA(Optimizer):
    def __init__(self, params, lr:float=required, cseq:dict={}, lam:float=1., regtype:str=None, iter:int=1, inertial:bool=False):
        if regtype not in [None, "l1", "l2", "elastic"]:
            raise ValueError("Invalid regularization method type: {}".format(regtype))
        if lr is not required and lr < 0.:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lam=lam, iter=1, previous_params=[])
        self.regtype = regtype
        self.inertial = inertial
        self.param_copy = []
        if 'alpha' in cseq.keys():
            self.alpha = lambda n : eval(cseq['alpha'])
        else: 
            self.alpha = lambda n : 1.
        if inertial:
            if 'theta' in cseq.keys():
                self.theta = lambda n : eval(cseq['theta'])
            else:
                self.theta = lambda n : 0.5**n
        super(FBA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FBA, self).__setstate__(state)
    
    def backward_part(self, x, c):
        if self.regtype == "l1":
            x = torch.sign(x)*torch.max(x - c, torch.zeros_like(x))
            return x
        elif self.regtype == "l2":
            return x/(c+1)
        elif self.regtype == "elastic": # (1-a)0.5|x|^2_2 + a|x|_1
            # TODO
            # x = x/(2-a)
            return x
        else:
            return x
    
    def inertial_step(self):
        for group in self.param_groups:
                if len(group['previous_params'])==0:
                    group['previous_params'] = [p.clone() for p in group['params']]
                for i, p in zip(range(len(group['params'])), group['params']):
                    dpq = p - group['previous_params'][i]
                    p.add_(dpq, alpha=self.theta(group['iter']))
                # update previous_params
                group['previous_params'] = []
                group['previous_params'] = [p.clone() for p in group['params']]
    
    @torch.no_grad()
    def update(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad
                # forward part
                tp = p.clone()
                tp.add_(dp, alpha=-group['lr'])
                # backward part
                tp = self.backward_part(tp, c=group['lr']*group['lam'])
                # step
                p.add_(p,alpha=-self.alpha(group['iter']))
                p.add_(tp, alpha=self.alpha(group['iter']))
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure

        self.update()
        if self.inertial:
            self.inertial_step()
        for group in self.param_groups:
            group['iter'] += 1
        return loss