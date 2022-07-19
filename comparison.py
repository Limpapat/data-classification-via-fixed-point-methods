from train import train
from experiment import experiment
from utils.utils import get_data
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import argparse

def compare(setting:str, istrain:bool=False):
    with open(setting, 'r') as f:
        data = json.loads(f.read())
        f.close()
    
    ### checking
    if 'data' not in data.keys():
        raise Exception("data not found, please assign your data")
    if not isinstance(data['data'], list):
        data['data'] = [data['data']]
    if 'model' not in data.keys():
        data['model'] = 'MLP'
    if 'loss' not in data.keys():
        data['loss'] = 'MSE'
    if 'penalty' not in data.keys():
        data['penalty'] = None
    if 'n_test' not in data.keys():
        data['n_test'] = .1
    if 'n_iter' not in data.keys():
        data['n_iter'] = 100
    if 'split_ratio' not in data.keys():
        data['split_ratio'] = [.1, .3]
    if 'disp' not in data.keys():
        data['disp'] = False
    if 'optimizers' not in data.keys():
        raise Exception("optimizers not found, please assign your optimizers")

    for d in data['data']:
        if 'path' not in d.keys():
            raise Exception("data path not found, please assign your data path")
        if 'name' not in d.keys():
            d['name'] = (d['path'].split('/')[-1]).split(".")[0]
        if 'feature_selection' not in d.keys():
            d['feature_selection'] = None
        if 'target' not in d.keys():
            d['target'] = -1
        data_path = d['path']
        data_name = d['name']
        data_label = d['class_label'] if 'class_label' in d.keys() else None
        data_feature_selection = d['feature_selection']
        data_target = d['target']
        print("=== START ===========================================================\n")
        print(f"DATASET: {data_name}")
        ### Load data
        X, y, label = get_data(csv_path=data_path, feature_selection=data_feature_selection, target=data_target, split_ratio=data['split_ratio'], disp=False, save_fig=True)
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        X_train, X_val, X_test = X
        y_train, y_val, y_test = y
        _n_classes = len(np.unique(np.concatenate((y_train, y_val, y_test), axis=None)))

        ### Convert array to tensor
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1).long()
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1).long()
        y_test = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1).long()
        
        ### Train & Experiment
        result_dict = {}
        for i, name in enumerate(data['optimizers'].keys()):
            opt = data['optimizers'][name]
            if 'lr' not in opt.keys():
                opt['lr'] = .2
            if 'rp' not in opt.keys():
                opt['rp'] = .01
            if 'cseq' not in opt.keys():
                opt['cseq'] = './cseq.json'
            args_save = f"comparison/{name}.pth"
            if istrain:
                train(data=data_path, 
                    feature_selection=data_feature_selection,
                    target=data_target,
                    split_ratio=data['split_ratio'],
                    disp=False,
                    args_loss=data['loss'],
                    args_optimizer=name,
                    penalty=data['penalty'],
                    reg_param=opt['rp'],
                    learning_rate=opt['lr'],
                    control_sequence_directory=opt['cseq'],
                    n_iteration=data['n_iter'],
                    args_save=args_save)
            
            acc_test, model, _LOSS, _ACCTRA, _ACCVAL = experiment(data=data_path,
                                                                    n_test=data['n_test'],
                                                                    trained_model=args_save,
                                                                    disp=False)
            result_dict[name] = {
                'acc_test' : acc_test,
                'model' : model,
                '_LOSS' : _LOSS,
                '_ACCTRA' : _ACCTRA,
                '_ACCVAL' : _ACCVAL
            }
            print("=== NEXT OPTIMIZER ==================================================\n")
        ### Display results
        _n_fig = len(data['optimizers'].keys()) + 1
        fig, axs = plt.subplots(3, _n_fig) if X_train.size()[-1] == 2 else plt.subplots(2, _n_fig)
        fig.set_figheight(10 if X_train.size()[-1]==2 else 7)
        fig.set_figwidth(_n_fig*4)
        for i, name in enumerate(data['optimizers'].keys()):
            acc_test, model, _LOSS, _ACCTRA, _ACCVAL = result_dict[name]['acc_test'], result_dict[name]['model'], result_dict[name]['_LOSS'], result_dict[name]['_ACCTRA'], result_dict[name]['_ACCVAL']
            axs[0, 0].plot(range(1, len(_LOSS)+1), _LOSS, label=name)
            axs[0, 0].set_xlabel('Iterations')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].set_title(f"ALL : Trained Model\'s Loss")
            axs[1, 0].plot(range(1, len(_ACCTRA)+1), _ACCTRA, label=name)
            axs[1, 0].set_xlabel('Iterations')
            axs[1, 0].set_ylabel('Accuracy')
            axs[1, 0].set_title(f"ALL : Trained Model\'s Accuracy")
        axs[0, 0].legend()
        axs[1, 0].legend()
        for i, name in enumerate(data['optimizers'].keys()):
            acc_test, model, _LOSS, _ACCTRA, _ACCVAL = result_dict[name]['acc_test'], result_dict[name]['model'], result_dict[name]['_LOSS'], result_dict[name]['_ACCTRA'], result_dict[name]['_ACCVAL']
            axs[0, i+1].plot(range(1, len(_LOSS)+1), _LOSS, marker='.')
            axs[0, i+1].set_xlabel('Iterations')
            axs[0, i+1].set_ylabel('Loss')
            axs[0, i+1].set_title(f"{name} : Trained Model\'s Loss")

            axs[1, i+1].plot(range(1, len(_ACCTRA)+1), _ACCTRA, label='train')
            axs[1, i+1].plot(range(1, len(_ACCVAL)+1), _ACCVAL, label='validation')
            axs[1, i+1].set_xlabel('Iterations')
            axs[1, i+1].set_ylabel('Accuracy')
            if X_train.size()[-1] == 2:
                axs[1, i+1].set_title(f"{name} : Trained Model\'s Accuracy")
            else:
                axs[1, i+1].set_title(f"{name} : Test Set Accuracy={acc_test*100:.2f}%")
            axs[1, i+1].legend()

            if X_train.size()[-1] == 2:
                label = label if data_label is None else data_label
                classifier=model.to(torch.device('cpu'))
                X = X_test.to(torch.device('cpu'))
                y = y_test.view(y_test.shape[0]).to(torch.device('cpu'))
                resolution = 0.02

                # setup marker generator and color map
                markers = ('s', 'o', '^', 'v')
                colors = ('red', 'blue', 'lightgreen', 'cyan')
                cmap = ListedColormap(colors[:len(np.unique(y))])

                # plot the decision surface
                x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                    np.arange(x2_min, x2_max, resolution))
                tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
                probas = classifier.forward(tensor)

                if i==0:
                    # plot test set
                    for idx, cl in enumerate(np.unique(y)):
                        axs[2, i].scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                                    alpha=0.8, color=cmap(idx),
                                    edgecolor='black',
                                    marker=markers[idx], 
                                    label=label[idx])
                    axs[2, i].grid(linestyle='--')
                    axs[2, i].set_title('Test set')
                    axs[2, i].set_xlim(xx1.min()-0.1, xx1.max()+0.1)
                    axs[2, i].set_ylim(xx2.min()-0.1, xx2.max()+0.1)
                    axs[2, i].legend()
                
                if probas.shape[1]==1:
                    tredshold = torch.zeros_like(probas).add_(0.5)
                    probas = torch.cat((probas, tredshold), 1)
                Z = np.argmax(probas.detach().numpy(), axis=1)

                Z = Z.reshape(xx1.shape)
                axs[2, i+1].contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
                axs[2, i+1].set_xlim(xx1.min()-0.1, xx1.max()+0.1)
                axs[2, i+1].set_ylim(xx2.min()-0.1, xx2.max()+0.1)

                # plot class samples
                for idx, cl in enumerate(np.unique(y)):
                    axs[2, i+1].scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                                alpha=0.8, color=cmap(idx),
                                edgecolor='black',
                                marker=markers[idx], 
                                label=label[idx])
                axs[2, i+1].set_title(f"{name} : Accuracy={acc_test*100:.2f}%")
                axs[2, i+1].legend()

            plt.tight_layout()
        plt.savefig(f"img/comparison_{data_name}_results.png")
        print("=== END =============================================================\n")
    if data['disp']:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--setting',
        type=str,
        default='./setting.json',
        help='Setting directory, default is \'./setting.json\''
    )
    args = parser.parse_args()
    compare(args.setting, istrain=True)