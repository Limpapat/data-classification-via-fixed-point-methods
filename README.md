# data-classification-via-fixed-point-methods
Experimentation on various iterative fixed point methods for data classification

# Train demo command
```
python train.py --data 'source/sample_generated_data.csv' --loss MCE
```
![This is an image](img/demo_trained_results.png)
# Experiment demo command
```
python experiment.py --test_data 'source/sample_generated_data.csv' --trained_model demomodel.pth --n_test .1
```
![This is an image](img/demo_experiment_results.png)
# Plan updated:
- Add forward-backward splitting optimizer (FBA)
- Add other FBA type opimizers i.e. ISFBA, PISFBA, IPFBA & related papers
- Write some documents
