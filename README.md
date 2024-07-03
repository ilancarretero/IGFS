## IGFS method implementation

It is structured in the following folders:

### Data

- Code related to the loading of data, partitions, seeds and variables to be selected.

### Features

- Multiple examples under different early stopping regimes to obtain the local attributions corresponding to the implemented deep learning model (autoencoder + classifier head) under a 5-fold CV framework.

- Implementation in the ```igfs_types``` folder of the attribute combination methods to extend Integrated Gradients to its global counterpart.

- In this folder would also be the implementations of the other deep learning based feature selection models. To use these implementations go to the authors' public repositories or request them via email.

### Models

- Implementation of hyperparameter optimization of the DL model using Optuna under different early stopping regimes.

- Implementation of the ML models used to evaluate variable selection using sklearn and MLflow to track the results.

### Visualization 

- Implementation of the visualizations made to analyze and compare results.