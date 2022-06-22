from model.model import MLP
from utils.utils import get_data, plot_decision_regions
import matplotlib.pyplot as plt
import torch
import argparse, os

def experiment(data:str='source/sample_generated_data.csv',
                n_test:float=.1,
                trained_model:str='demomodel.pth',
                disp=False):
    print("\n========= INITAILIZING =========\n")
    print(f"Data directory : \'{data}\'")

    ### Get data
    if os.path.isfile(data):
        X, y = get_data(csv_path=data, split_ratio=[n_test], disp=False)
    else:
        raise ValueError('Invalid data direcory')
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    ### Convert array to tensor
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1).long()

    ### Load trained model
    model_saved_path = 'trained_models'
    args_checkpoint = os.path.join(model_saved_path, trained_model)
    if os.path.isfile(args_checkpoint):
        checkpoint = torch.load(args_checkpoint, map_location=device)
        print('Trained model: ', checkpoint['name'])
        ### Set up model
        _n_features, _n_classes = X_test.shape[-1], torch.unique(y_test).shape[0]
        model = MLP(num_features=_n_features, num_classes=_n_classes, activation='softmax')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError('Invalid name : trained model file name {} not found on {}'.format(trained_model, model_saved_path))

    ### Experiment results
    print("\n====== EXPERIMENT RESULTS ======\n")
    model.eval()
    probas = model(X_test)
    score = torch.where(torch.argmax(probas, dim=1) == y_test, 1, 0).sum()
    acc_test = (score/probas.shape[0]).item()
    print('Predicted score: {}/{}'.format(score, probas.shape[0]))
    print('Accuracy: ', acc_test)
    print("================================\n")

    ### Display results
    if disp:
        _LOSS, _ACCTRA, _ACCVAL = checkpoint['loss'], checkpoint['train_acc'], checkpoint['validation_acc']
        label = [0, 1, 2]
        plt.figure(figsize=(15,4))
        plt.subplot(131)
        plot_decision_regions(X_test.to(torch.device('cpu')), 
                            y_test.view(y_test.shape[0]).to(torch.device('cpu')), 
                            classifier=model.to(torch.device('cpu')), 
                            label=label)
        plt.title(f"Experiment Results : accuracy={acc_test*100:.2f}%")
        plt.legend()

        plt.subplot(132)
        plt.plot(range(1, len(_LOSS)+1), _LOSS, marker='.')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Milti-classes Classification : Trained Model\'s Loss')

        plt.subplot(133)
        plt.plot(range(1, len(_ACCTRA)+1), _ACCTRA, label='train')
        plt.plot(range(1, len(_ACCVAL)+1), _ACCVAL, label='validation')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Milti-classes Classification : Trained Model\'s Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('img/experiment_results.png')
        plt.show()
    return acc_test, model, _LOSS, _ACCTRA, _ACCVAL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--test_data',
        type=str,
        default='',
        help='Dataset directory'
    )
    parser.add_argument(
        '--disp',
        action="store_true",
        help='Display option to show experiment results, default is False'
    )
    parser.add_argument(
        '-n', '--n_test',
        type=float,
        default=1.,
        help='Number of test data, default is 1. which means get test 100%% data of dataset'
    )
    parser.add_argument(
        '-m', '--trained_model',
        type=str,
        default='demomodel.pth',
        help='A trained model\'s name, default is \'demomodel.pth\''
    )

    args = parser.parse_args()
    experiment(data=args.test_data,
                n_test=args.n_test,
                trained_model=args.trained_model,
                disp=args.disp)
    