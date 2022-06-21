from torch.optim.optimizer import Optimizer, required
from math import sqrt
import torch
import threading

"""
    .-------- Class Structure --------.
    |               FBA               |
    |                |                |
    |               SFBA              |
    |                |                |
    |          .-----------.          |
    |          |           |          |
    |    ParallelSFBA     PFBA        |
    .---------------------------------.
"""

class FBA(Optimizer):
    """
        Forward Backward Algorithm (FBA) is a base calss for forward backward splitting type algorithms
            Parameters :
                params : model's parameters
                lr : learning rate : a positive real number
                cseq : control sequence : a dictionary of control sequence parameters, default is {}
                lam : regularization parameter : a positive real number, default is 1.
                regtype : regularization method type viz : l1 (L1-regularization), l2 (L2-regularization), elastic (Elastic-Net-regularization)
                iter : initial iteration step : an integer number, default is 1
                inertial : inertial step : boolean, default is False
                registered_tp : registered T(x) for one-more step iteration such as S-iteration : boolean, default is False
            .: Math::
                y_n = x_n + theta_n * (x_n - x_{n-1})                   ... [ if inertial = True ]
                x_{n+1} = (1 - alpha_n) * x_n + alpha_n * T(y_n)
                where T(x) = PROX_lr*lam*G(x - lr*grad(x)), and PROX_f is a proximity operator of f
            .: Ref. book:: (Chapter 28) <https://books.google.co.th/books/about/Convex_Analysis_and_Monotone_Operator_Th.html?id=cxL3jL7ONjQC&redir_esc=y>
    """
    def __init__(self, params, 
                    lr:float=required, 
                    cseq:dict={}, 
                    lam:float=1., 
                    regtype:str=None, 
                    inititer:int=1, 
                    inertial:bool=False):
        if regtype not in [None, "l1", "l2", "elastic"]:
            raise ValueError("Invalid regularization method type: {}".format(regtype))
        if lr is not required and lr < 0.:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lam=lam)
        super(FBA, self).__init__(params, defaults)
        self.state = {'iter' : inititer}
        self.regtype = regtype
        self.inertial = inertial
        self.registered_tp = False
        self.parallel = False
        if 'alpha' in cseq.keys():
            self.alpha = lambda n : eval(cseq['alpha'])
        else: 
            self.alpha = lambda n : 1.
        if inertial:
            self.state['previous_params'] = [[] for group in self.param_groups]
            if 'theta' in cseq.keys():
                self.theta = lambda n : eval(cseq['theta'])
            else:
                self.theta = lambda n : sqrt(1+4*n*n)/(1+sqrt(1+4*(n+1)*(n+1)))
    
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
    
    @torch.no_grad()
    def inertial_step(self):
        state = self.state
        state_previous_params = state['previous_params']
        for gi, group in zip(range(len(self.param_groups)), self.param_groups):
                if len(state_previous_params[gi])==0:
                    state_previous_params[gi] = [p.clone() for p in group['params']]
                for i, p in zip(range(len(group['params'])), group['params']):
                    dpq = p - state_previous_params[gi][i]
                    if self.parallel:
                        y = p.clone()
                        y.add_(dpq, alpha=self.theta(state['iter']))
                        self.state['y'][gi].append(y)
                    else:
                        p.add_(dpq, alpha=self.theta(state['iter']))
                # update previous_params
                self.state['previous_params'][gi] = [p.clone() for p in group['params']]
    
    @torch.no_grad()
    def update(self, registered_tp=False, cseq=None):
        if cseq is None:
            cseq = self.alpha
        state = self.state
        for gi, group in zip(range(len(self.param_groups)), self.param_groups):
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad
                # forward part
                tp = p.clone()
                tp.add_(dp, alpha=-group['lr'])
                # backward part
                tp = self.backward_part(tp, c=group['lr']*group['lam'])
                if registered_tp:
                    # store tp for next step
                    self.state['tp'][gi].append(tp.clone())
                if self.parallel:
                    z = p.clone()
                    z.add_(z, alpha=-cseq(state['iter']))
                    z.add_(tp, alpha=cseq(state['iter']))
                    self.state['z'][gi].append(z)
                else:
                    # step
                    p.add_(p,alpha=-cseq(state['iter']))
                    p.add_(tp, alpha=cseq(state['iter']))
    
    def step(self, closure=None):
        state = self.state
        if self.inertial:
            self.inertial_step()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.update()
        state['iter'] += 1
        return loss

class SFBA(FBA):
    """
        S-Forward Backward Algorithm (SFBA) is an algorithm based on forward backward splitting algorithms.
            Parameters:
                params : model's parameters
                lr : learning rate : a positive real number
                cseq : control sequence : a dictionary of control sequence parameters, default is {}
                lam : regularization parameter : a positive real number, default is 1.
                regtype : regularization method type viz : l1 (L1-regularization), l2 (L2-regularization), elastic (Elastic-Net-regularization)
                iter : initial iteration step : an integer number, default is 1
                inertial : inertial step : boolean, default is False
                registered_tp : registered T(x) for one-more step iteration such as S-iteration : boolean, default is False
            .: Math::
                y_n = x_n + theta_n * (x_n - x_{n-1})                   ... [ if inertial = True ]
                z_n = (1 - alpha_n) * y_n + alpha_n * T(y_n)
                x_{n+1} = (1 - beta_n) * T(y_n) + beta_n * T(z_n)
                where T(x) = PROX_lr*lam*G(x - lr*grad(x)), and PROX_f is a proximity operator of f
            .: Ref. paper:: <http://www.doiserbia.nb.rs/img/doi/0354-5180/2021/0354-51802103771B.pdf>
    """
    def __init__(self, params, 
                    lr:float=required, 
                    cseq:dict={}, 
                    lam:float=1., 
                    regtype:str=None, 
                    inititer:int=1, 
                    inertial:bool=False):
        if 'beta' in cseq.keys():
            self.beta = lambda n : eval(cseq['beta'])
        else: 
            self.beta = lambda n : 1.
        super(SFBA, self).__init__(params, lr, cseq, lam, regtype, inititer, inertial)
        self.registered_tp = True
        if self.registered_tp:
            self.state['tp'] = [[] for group in self.param_groups]

    @torch.no_grad()
    def finalupdate(self):
        """
            final-step
        """
        state = self.state
        state_ty = state['tp']
        for gi, group in zip(range(len(self.param_groups)), self.param_groups):
            for i, p in zip(range(len(group['params'])), group['params']):
                if p.grad is None:
                    continue
                dp = p.grad
                # forward part
                tp = p.clone()
                tp.add_(dp, alpha=-group['lr'])
                # backward part
                tp = self.backward_part(tp, c=group['lr']*group['lam'])
                # final-step
                ty = state_ty[gi][i]
                p.mul_(torch.zeros_like(p)).add_(ty)
                p.add_(p,alpha=-self.beta(state['iter']))
                p.add_(tp, alpha=self.beta(state['iter']))
        # reset state tp
        self.state['tp'] = [[] for group in self.param_groups]

    def step(self, closure=None):
        state = self.state
        if self.inertial:
            self.inertial_step()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.update(registered_tp=self.registered_tp)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.finalupdate()

        state['iter'] += 1
        return loss

class ParallelSFBA(SFBA):
    """
        Parallel S-Forward Backward Algorithm (Parallel-SFBA) is an algorithm based on forward backward splitting algorithms.
            Parameters:
                params : model's parameters
                lr : learning rate : a positive real number
                cseq : control sequence : a dictionary of control sequence parameters, default is {}
                lam : regularization parameter : a positive real number, default is 1.
                regtype : regularization method type viz : l1 (L1-regularization), l2 (L2-regularization), elastic (Elastic-Net-regularization)
                iter : initial iteration step : an integer number, default is 1
                inertial : inertial step : boolean, default is False
                registered_tp : registered T(x) for one-more step iteration such as S-iteration : boolean, default is False
            .: Math::
                y_n = x_n + theta_n * (x_n - x_{n-1})                   ... [ if inertial = True ]
                z_n = (1 - alpha_n) * x_n + alpha_n * T(x_n)
                x_{n+1} = (1 - beta_n) * T(y_n) + beta_n * T(z_n)
                where T(x) = PROX_lr*lam*G(x - lr*grad(x)), and PROX_f is a proximity operator of f
            .: Ref. paper:: <https://www.carpathian.cunbm.utcluj.ro/wp-content/uploads/2020_vol_36_1/carpathian_2020_36_1_35_44_abstract.pdf>
    """
    def __init__(self, params, 
                    lr:float=required, 
                    cseq:dict={}, 
                    lam:float=1., 
                    regtype:str=None, 
                    inititer:int=1, 
                    inertial:bool=False):
        if 'beta' in cseq.keys():
            self.beta = lambda n : eval(cseq['beta'])
        else: 
            self.beta = lambda n : 1.
        super(ParallelSFBA, self).__init__(params, lr, cseq, lam, regtype, inititer, inertial)
        self.registered_tp = True
        self.state['y'] = [[] for group in self.param_groups]
        self.state['z'] = [[] for group in self.param_groups]
        self.parallel = True
    
    def updatemodelparameters(self, from_state):
        for gi, group in zip(range(len(self.param_groups)), self.param_groups):
            for i, p in zip(range(len(group['params'])), group['params']):
                p.mul_(torch.zeros_like(p)).add_(from_state[gi][i])

    @torch.no_grad()
    def parallelstep(self, closure):
        """
            parallel-step
        """
        state = self.state
        with torch.enable_grad():
            loss = closure()
        parallel_z = threading.Thread(target=self.update, args=[False, None])
        parallel_z.start()
        if self.inertial:
            parallel_y = threading.Thread(target=self.inertial_step)
            parallel_y.start()
            parallel_y.join()
        parallel_z.join()
        ### compute Ty
        self.updatemodelparameters(from_state=state['y'])
        with torch.enable_grad():
            loss = closure()
        for gi, group in zip(range(len(self.param_groups)), self.param_groups):
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad
                # forward part
                tp = p.clone()
                tp.add_(dp, alpha=-group['lr'])
                # backward part
                tp = self.backward_part(tp, c=group['lr']*group['lam'])
                self.state['tp'][gi].append(tp.clone())
        # reset state y
        self.state['y'] = [[] for group in self.param_groups]
        ### update model parameters to z
        self.updatemodelparameters(from_state=state['z'])

    def step(self, closure=None):
        state = self.state
        if closure is not None:
            self.parallelstep(closure)
        else:
            raise ValueError(f"Closure is {closure}. A parallel method need some closure to compute step, please assign closure")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.finalupdate()
        # reset state z
        self.state['z'] = [[] for group in self.param_groups]

        state['iter'] += 1
        return loss

class PFBA(SFBA):
    """
        P-Forward Backward Algorithm (PFBA) is an algorithm based on forward backward splitting algorithms.
            Parameters:
                params : model's parameters
                lr : learning rate : a positive real number
                cseq : control sequence : a dictionary of control sequence parameters, default is {}
                lam : regularization parameter : a positive real number, default is 1.
                regtype : regularization method type viz : l1 (L1-regularization), l2 (L2-regularization), elastic (Elastic-Net-regularization)
                iter : initial iteration step : an integer number, default is 1
                inertial : inertial step : boolean, default is False
                registered_tp : registered T(x) for one-more step iteration such as S-iteration : boolean, default is False
            .: Math::
                y_n = x_n + theta_n * (x_n - x_{n-1})                   ... [ if inertial = True ]
                z_n = (1 - alpha_n) * y_n + alpha_n * T(y_n)
                u_n = (1 - gamma_n) * z_n + gamma_n * T(z_n)
                x_{n+1} = (1 - beta_n) * T(z_n) + beta_n * T(u_n)
                where T(x) = PROX_lr*lam*G(x - lr*grad(x)), and PROX_f is a proximity operator of f
            .: Ref. paper:: <http://thaijmath.in.cmu.ac.th/index.php/thaijmath/article/viewFile/3856/354354774>
    """
    def __init__(self, params, 
                    lr:float=required, 
                    cseq:dict={}, 
                    lam:float=1., 
                    regtype:str=None, 
                    inititer:int=1, 
                    inertial:bool=False):
        if 'beta' in cseq.keys():
            self.beta = lambda n : eval(cseq['beta'])
        else: 
            self.beta = lambda n : 1.
        if 'gamma' in cseq.keys():
            self.gamma = lambda n : eval(cseq['gamma'])
        else: 
            self.gamma = lambda n : 1.
        super(PFBA, self).__init__(params, lr, cseq, lam, regtype, inititer, inertial)
        self.registered_tp = True

    def step(self, closure=None):
        state = self.state
        if self.inertial:
            self.inertial_step()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.update()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.update(registered_tp=self.registered_tp, cseq=self.gamma)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.finalupdate()
        
        state['iter'] += 1
        return loss