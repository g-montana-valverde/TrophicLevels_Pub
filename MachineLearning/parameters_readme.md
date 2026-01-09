# Configuration Parameters

This file describes the parameters used in `parameters.json` for your data processing and model training pipeline.

---

### General Settings

- **TEST_SIZE** (`float`):  
  Proportion of the dataset to include in the test split.  
  _Example_: `0.25` → 25% test data.

- **DO_KFOLD_TEST** (`bool`): Whether to perform a K-fold approach to the train/test 
  split to report results avoiding single train/test partition variability. K will come 
  determined by **TEST_SIZE**: **K = int(1/TEST_SIZE)**

- **RANDOM_STATE** (`int`):  
  Seed used by the random number generator for reproducibility.  
  _Example_: `2` 

- **GROUPS** (`dict`): Dictionary with the key-value pairs for the groups and
    numerical label associated to each group.
    _Example_: ``{"HC": 0, "AD": 1}``

- - **ID_key** (`string`): Name of the column in the excel file where the
    participant ID is stored.
    _Example_: ``"PTID"``

- **MODEL_TYPE** (`string`): Model type to be used. Currently implemented model types 
  (classifiers) are: "SVC", "LogReg", "KNNClassifier"

- **MODEL_KWARGS** (`dict`): Dictionary with additional kwargs to pass to the model 
  definition. These are model parameters that are fixed beforehand that will not be 
  explored during the hyperparameter search. The dict keys must have the same string 
  value as the names of the arguments to initialize the model.

- **NF** (`int`): Fixed Number of Features to be used with MRMR selection. NF can 
  also be passed as a Hyperparameter to be explored, in that case, this "NF" can be ignored.

- **USE_MRMR_FEATURE_SELECTION** (`bool`):  
  Enables MRMR feature selection algorithm.
 
---

### Grid Search Parameters (`GRID_SEARCH`)

- **HYPERPARAM_SWEEP** (`dict`): dict where each key is equal to the model parameter 
  to be explored, and the values are an iterable of values of that parameter to 
  explore. For instance, for an SVC: ``{"kernel": ["rbf", "poly"], "C": [0.1, 10]}``.
  If the hyperparameter ``NF`` is passed, then NF features are selected is one of 
  the two hyperparameters explored.
- **SCORE** (`string`): The metric score that is used to evaluate which 
  hyperparameter combination is the best. It can be either "AUC", "accuracy", or 
  "bal_accuracy".

---

> Modify `parameters.json` to adjust model behavior.
