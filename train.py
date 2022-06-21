from model.model import MLP
from utils.utils import get_data, plot_decision_regions, loadcseq
from utils.functional import *
from datetime import datetime
from optim.optim import FBA, SFBA, PFBA, ParallelSFBA
import matplotlib.pyplot as plt
import argparse
import torch
import os, sys, time
import torch.optim as optim

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
        '--disp',
        action="store_true",
        help='Display option to show dataset, default is False'
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
    ### plan ###
    # add more argument

    args = parser.parse_args()
    print("\n========= INITAILIZING =========\n")
    print(f"Data directory : \'{args.data}\'")
    if os.path.isfile(args.data):
        X, y = get_data(csv_path=args.data, split_ratio=args.split_ratio, disp=args.disp)
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
    print(f"Loss : \'{args.loss}\'")
    if args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    elif args.loss == 'BCE':
        loss_fn = torch.nn.BCEloss()
    elif args.loss == 'MCE':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unsupported loss : loss should be in [\'MSE\', \'BCE\', \'MCE\']')
    
    if args.penalty is not None:
        print(f"Regularization : \'{args.penalty}\'")
        print(f"Regularization parameter : \'{args.reg_param}\'")
        if args.penalty == 'l1':
            loss_reg_fn = l1_fn
        elif args.penalty == 'l2':
            loss_reg_fn = l2_fn
        else:
            raise ValueError('Unsupported penalty : penalty should be in [\'l1\', \'l2\', \'None\']')

    ### Setting model & optimizer
    _n_features, _n_classes = X_train.shape[-1], torch.unique(y_train).shape[0]
    # torch.manual_seed(0)
    model = MLP(num_features=_n_features, num_classes=_n_classes, activation='softmax')
    print(f"Optimizer: \'{args.optimizer}\'")
    print(f"\tLearning rate: {args.learning_rate}")
    if args.control_sequence_directory is not None and args.optimizer != 'SGD':
        cseq = loadcseq(args.control_sequence_directory)
        print("\tControl sequence parameters: ")
        for k in cseq.keys():
            print(f"\t\t{k}: {cseq[k]}")
    else:
        cseq = {}
    if args.optimizer == 'SGD':
        _splitting_type_method = False
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'FBA':
        _splitting_type_method = True
        optimizer = FBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args.optimizer == 'IFBA':
        _splitting_type_method = True
        optimizer = FBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=True)
    elif args.optimizer == 'SFBA':
        _splitting_type_method = True
        optimizer = SFBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args.optimizer == 'ParallelSFBA':
        _splitting_type_method = True
        optimizer = ParallelSFBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args.optimizer == 'PFBA':
        _splitting_type_method = True
        optimizer = PFBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=False)
    elif args.optimizer == 'ISFBA':
        _splitting_type_method = True
        optimizer = SFBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=True)
    elif args.optimizer == 'ParallelISFBA':
        _splitting_type_method = True
        optimizer = ParallelSFBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=True)
    elif args.optimizer == 'IPFBA':
        _splitting_type_method = True
        optimizer = PFBA(model.parameters(), 
                        lr=args.learning_rate, 
                        lam=args.reg_param, 
                        regtype=args.penalty, 
                        cseq=cseq,
                        inertial=True)
    else:
        raise ValueError(f"Invalid optimizer : optimizer {args.optimizer} not found : please check our optimizer supported")

    print("\n=========== TRAINING ===========\n")
    _LOSS, _ACCTRA, _ACCVAL = [], [], []
    start_time = time.time()
    for epoch in range(args.n_iteration):
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
            loss_reg = sum(loss_reg_fn(p.detach()) for p in model.parameters()) if args.penalty is not None else 0.
            loss += args.reg_param * loss_reg
            _LOSS.append(loss.item())
        else:
            loss_reg = sum(loss_reg_fn(p) for p in model.parameters()) if args.penalty is not None else 0.
            loss = loss_fn(probas, y_train) + args.reg_param * loss_reg
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
            print('Epoch: %03d/%03d' % ((epoch+1), args.n_iteration), end="")
            print(' | Train acc: %.2f' % (_ACCTRA[-1]), end="")
            print(' | Loss: %.3f' % _LOSS[-1], end="")
            print(' | Val acc: %.2f' % (_ACCVAL[-1]))
    
    elapsed_time = time.time() - start_time
    print("\nElapsed time: %.2fs" % (elapsed_time))
    print("================================\n")

    ### Save model
    if args.save is not None:
        save_path = 'trained_models'
        name_save = args.save + ".pth" if args.save.split('.')[-1] != 'pth' else name_save
        checkpoint_dict = {
            'n_iters' : args.n_iteration,
            'name' : name_save,
            'time_created' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_state_dict' : model.state_dict(),
            'optimizer' : args.optimizer,
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : _LOSS,
            'train_acc' : _ACCTRA,
            'validation_acc' : _ACCVAL,
            'elapsed_time' : elapsed_time
        }
        if args.penalty is not None:
            checkpoint_dict['penalty'] = args.penalty
            checkpoint_dict['reg_param'] = args.reg_param
        torch.save(checkpoint_dict, os.path.join(save_path,name_save))

    if args.disp:
        label = [0, 1, 2]
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




