from model.model import MLP
from utils.utils import get_data, plot_decision_regions
import matplotlib.pyplot as plt
import argparse
import torch
import os, sys, time

import torch.optim as optim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_iteration',
        type=int,
        default=100,
        help='Number of iteration, default is 100'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='',
        help='Dataset directory'
    )
    parser.add_argument(
        '--disp',
        type=bool,
        default=False,
        help='Display option to show dataset, default is False'
    )
    parser.add_argument(
        '--split_ratio',
        type=list,
        default=[.1, .3],
        help='List of ratio to split dataset, default is [.1, .3] which means get test 10%% data of dataset and get validation 30%% data of remains'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='MSE',
        help='Loss function, default is \'MSE\''
    )
    ### plan ###
    # add more argument

    args = parser.parse_args()
    print("\n===== INITAILIZING =====\n")
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

    ### Setting model & optimizer
    _n_features, _n_classes = X_train.shape[-1], torch.unique(y_train).shape[0]
    model = MLP(num_features=_n_features, num_classes=_n_classes, activation='softmax')
    optimizer = optim.SGD(model.parameters(), lr=.2)
    print("Optimizer: SGD")

    print("\n===== TRAINING =====\n")
    _LOSS, _ACCTRA, _ACCVAL = [], [], []
    start_time = time.time()
    for epoch in range(args.n_iteration):
        ### train
        model.train()
        probas = model(X_train)
        score = torch.where(torch.argmax(probas, dim=1) == y_train, 1, 0).sum()
        _ACCTRA.append(score/probas.shape[0])
        loss = loss_fn(probas, y_train)
        _LOSS.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    print("===================\n")

    #### Update later! ####
    label = [0,1, 2]
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
    plt.title('Loss')

    plt.subplot(224)
    plt.plot(range(1, len(_ACCTRA)+1), _ACCTRA, label='train')
    plt.plot(range(1, len(_ACCVAL)+1), _ACCVAL, label='validation')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Milti-classes Classification : Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()




