from dcfp.model.model import MLP
from dcfp.utils.utils import get_data, plot_decision_regions, loadcseq
from dcfp.utils.functional import *
from datetime import datetime
from dcfp.optim.optim import FBA, SFBA, PFBA, ParallelSFBA
import matplotlib.pyplot as plt
import argparse
import torch
import os, time
import torch.optim as optim

def train(data:str='source/sample_generated_data.csv', 
        feature_selection:list=None,
        target:int=-1,
        split_ratio:list=[.1, .3],
        disp:bool=False,
        args_loss:str='MCE',
        args_optimizer:str='FBA',
        penalty:str='l1',
        reg_param:float=1e-3,
        learning_rate:float=2.,
        control_sequence_directory:str='./cseq.json',
        n_iteration:int=100,
        args_save:str=None,
        save_fig:bool=False):
    print("\n========= INITAILIZING =========\n")
    print(f"Data directory : \'{data}\'")
    if os.path.isfile(data):
        X, y, label = get_data(csv_path=data, feature_selection=feature_selection, target=target, split_ratio=split_ratio, disp=disp, save_fig=save_fig)
    else:
        raise ValueError('Invalid data direcory')
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    ### Convert array to tensor
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1).long()
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1).long()
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1).long()

    ### Setting loss
    print(f"Loss : \'{args_loss}\'")
    if args_loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif args_loss == 'BCE':
        loss_fn = torch.nn.BCEloss()
    elif args_loss == 'MCE':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unsupported loss : loss should be in [\'MSE\', \'BCE\', \'MCE\']')
    
    if penalty is not None:
        print(f"Regularization : \'{penalty}\'")
        print(f"Regularization parameter : \'{reg_param}\'")
        if penalty == 'l1':
            loss_reg_fn = l1_fn
        elif penalty == 'l2':
            loss_reg_fn = l2_fn
        else:
            raise ValueError('Unsupported penalty : penalty should be in [\'l1\', \'l2\', \'None\']')

    ### Setting model & optimizer
    _n_features, _n_classes = X_train.shape[-1], torch.unique(y_train).shape[0]
    # torch.manual_seed(0)
    model = MLP(num_features=_n_features, num_classes=_n_classes, activation='softmax')
    print(f"Optimizer: \'{args_optimizer}\'")
    print(f"\tLearning rate: {learning_rate}")
    if control_sequence_directory is not None and args_optimizer != 'SGD':
        cseq = loadcseq(control_sequence_directory)
        print("\tControl sequence parameters: ")
        for k in cseq.keys():
            print(f"\t\t{k}: {cseq[k]}")
    else:
        cseq = {}
    if args_optimizer == 'SGD':
        _splitting_type_method = False
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif args_optimizer == 'FBA':
        _splitting_type_method = True
        optimizer = FBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args_optimizer == 'IFBA':
        _splitting_type_method = True
        optimizer = FBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=True)
    elif args_optimizer == 'SFBA':
        _splitting_type_method = True
        optimizer = SFBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args_optimizer == 'ParallelSFBA':
        _splitting_type_method = True
        optimizer = ParallelSFBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args_optimizer == 'PFBA':
        _splitting_type_method = True
        optimizer = PFBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args_optimizer == 'ISFBA':
        _splitting_type_method = True
        optimizer = SFBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=True)
    elif args_optimizer == 'ParallelISFBA':
        _splitting_type_method = True
        optimizer = ParallelSFBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=True)
    elif args_optimizer == 'IPFBA':
        _splitting_type_method = True
        optimizer = PFBA(model.parameters(), 
                        lr=learning_rate, 
                        lam=reg_param, 
                        regtype=penalty, 
                        cseq=cseq,
                        inertial=True)
    else:
        raise ValueError(f"Invalid optimizer : optimizer {args_optimizer} not found : please check our optimizer supported")

    print("\n=========== TRAINING ===========\n")
    _LOSS, _ACCTRA, _ACCVAL = [], [], []
    start_time = time.time()
    for epoch in range(n_iteration):
        # for var_name in optimizer.state_dict():
        #     print(var_name, '\t', optimizer.state_dict()[var_name])
        ### train
        model.train()
        probas = model(X_train)
        score = torch.where(torch.argmax(probas, dim=1) == y_train, 1, 0).sum()
        _ACCTRA.append(score/probas.shape[0])
        if _splitting_type_method:
            # --------------------------------------------------
            def closure():
                optimizer.zero_grad()
                probas = model(X_train)
                loss = loss_fn(probas, y_train)
                loss.backward()
                return loss
            # --------------------------------------------------
            optimizer.step(closure)
            loss = loss_fn(probas, y_train)
            loss_reg = sum(loss_reg_fn(p.detach()) for p in model.parameters()) if penalty is not None else 0.
            loss += reg_param * loss_reg
            _LOSS.append(loss.item())
        else:
            loss_reg = sum(loss_reg_fn(p) for p in model.parameters()) if penalty is not None else 0.
            loss = loss_fn(probas, y_train) + reg_param * loss_reg
            _LOSS.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            x = optimizer.step()
        ### validation
        with torch.no_grad():
            model.eval()
            output = model(X_val)
            score = torch.where(torch.argmax(output, dim=1) == y_val, 1, 0).sum()
            _ACCVAL.append(score/output.shape[0])
        if (epoch+1) % 10 == 0:
            print('Epoch: %03d/%03d' % ((epoch+1), n_iteration), end="")
            print(' | Train acc: %.2f' % (_ACCTRA[-1]), end="")
            print(' | Loss: %.3f' % _LOSS[-1], end="")
            print(' | Val acc: %.2f' % (_ACCVAL[-1]))
    
    elapsed_time = time.time() - start_time
    print("\nElapsed time: %.2fs" % (elapsed_time))
    print("================================\n")

    ### Save model
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    if args_save is not None:
        save_path = 'trained_models'
        name_save = args_save + ".pth" if args_save.split('.')[-1] != 'pth' else args_save
        checkpoint_dict = {
            'n_iters' : n_iteration,
            'name' : name_save,
            'feature_selection' : feature_selection,
            'target' : target,
            'time_created' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_state_dict' : model.state_dict(),
            'optimizer' : args_optimizer,
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : _LOSS,
            'train_acc' : _ACCTRA,
            'validation_acc' : _ACCVAL,
            'elapsed_time' : elapsed_time
        }
        if penalty is not None:
            checkpoint_dict['penalty'] = penalty
            checkpoint_dict['reg_param'] = reg_param
        torch.save(checkpoint_dict, os.path.join(save_path,name_save))

    if disp:
        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plot_decision_regions(X_train.to(torch.device('cpu')), 
                            y_train.view(y_train.shape[0]).to(torch.device('cpu')), 
                            classifier=model.to(torch.device('cpu')), 
                            label=label)
        plt.title('Train Set')
        plt.legend()

        plt.subplot(222)
        plot_decision_regions(X_val.to(torch.device('cpu')), 
                            y_val.view(y_val.shape[0]).to(torch.device('cpu')), 
                            classifier=model.to(torch.device('cpu')), 
                            label=label)
        plt.title('Validation Set')
        plt.legend()

        plt.subplot(223)
        plt.plot(range(1, len(_LOSS)+1), _LOSS, marker='.')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Milti-classes Classification : Loss')

        plt.subplot(224)
        plt.plot(range(1, len(_ACCTRA)+1), _ACCTRA, label='train')
        plt.plot(range(1, len(_ACCVAL)+1), _ACCVAL, label='validation')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Milti-classes Classification : Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('img/trained_results.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--n_iteration',
        type=int,
        default=100,
        help='Number of iteration, default is 100'
    )
    parser.add_argument(
        '-d', '--data',
        type=str,
        default='',
        help='Dataset directory'
    )
    parser.add_argument(
        '-fs', '--feature_selection',
        type=int,
        nargs='+',
        default=[0, 1],
        help='List of feature indexes of trained dataset, default is [0, 1]'
    )
    parser.add_argument(
        '-t', '--target',
        type=int,
        default=-1,
        help='An index of target/label of trained dataset, default is -1'
    )
    parser.add_argument(
        '--disp',
        action="store_true",
        help='Display option to show dataset, default is False'
    )
    parser.add_argument(
        '--not_save_data_fig',
        action="store_false",
        help='Save dataset figure option, default is True'
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        nargs='+',
        default=[.1, .3],
        help='List of ratio to split dataset, default is [.1, .3] which means get test 10%% data of dataset and get validation 30%% data of remains'
    )
    parser.add_argument(
        '-csd', '--control_sequence_directory',
        type=str,
        default=None,
        help='A control sequences info directory, i.e. \'./cseq.json\', default is None'
    )
    parser.add_argument(
        '-l', '--loss',
        type=str,
        default='MSE',
        help='Loss function, default is \'MSE\''
    )
    parser.add_argument(
        '-s', '--save',
        type=str,
        default=None,
        help='Saved model\'s name, default is None. The trained model didn\'t be saved if --save is None'
    )
    parser.add_argument(
        '-o', '--optimizer',
        type=str,
        default='SGD',
        help='An optimizer name, default is \'SGD\''
    )
    parser.add_argument(
        '-p', '--penalty',
        type=str,
        default=None,
        help='Regularization method, default is None. --penalty sould be in [\'l1\', \'l2\', \'None\']'
    )
    parser.add_argument(
        '-rp', '--reg_param',
        type=float,
        default=0.01,
        help='Regularization parameter, default is 0.01'
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=0.2,
        help='Learning rate, default is 0.2'
    )

    args = parser.parse_args()
    train(data=args.data, 
        feature_selection=args.feature_selection,
        split_ratio=args.split_ratio,
        target=args.target,
        disp=args.disp,
        args_loss=args.loss,
        args_optimizer=args.optimizer,
        penalty=args.penalty,
        reg_param=args.reg_param,
        learning_rate=args.learning_rate,
        control_sequence_directory=args.control_sequence_directory,
        n_iteration=args.n_iteration,
        args_save=args.save,
        save_fig=args.not_save_data_fig)