# data-classification-via-fixed-point-methods
Experimentation on various iterative fixed point methods for data classification

# What's New:
- forward-backward splitting optimizer (FBA) & inertial FBA optimizer (IFBA)

# TODO:
- Add other FBA type opimizers i.e. ISFBA, PISFBA, IPFBA & related papers
- Write some documents

# Train demo command
```
python train.py --data 'source/sample_generated_data.csv' --loss MCE
```
# Experiment demo command
```
python experiment.py --test_data 'source/sample_generated_data.csv' --trained_model demomodel.pth --n_test .1
```
# Experimantal results
- without Regularization:
![This is an image](img/demo_trained_results.png)
![This is an image](img/demo_experiment_results.png)
- L1-Regularization:
![This is an image](img/l1_trained_results.png)
![This is an image](img/l1_experiment_results.png)
- L2-Regularization:
![This is an image](img/l2_trained_results.png)
![This is an image](img/l2_experiment_results.png)
