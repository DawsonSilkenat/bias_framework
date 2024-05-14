# Introduction

The bias framework is designed to provide insight into bias and bias mitigation in machine learning. The goal is not to solve the problem or even to make a recommendation, but instead to provide a user with information that will help them make the required decisions.  

Please note that this is still a work in progress, with a number of limitations. At th time of writing this (2024/05/13) we limit the bias framework to sklearn clarification models. Given time we would love to increase the flexibility and perhaps even increase the scope beyond classification. We also hope that this will never be truly complete as further research introduces innovative techniques for measuring and reducing bias.

## High level overview
A user will need to provide the bias framework with a machine model and a training/validation data split. They will then be able to define what they consider to be a privileged group, then simply call the run_framework method. After this method complete the user will be able to use a number of methods to generate and combine graphs displaying the impacts of bias across a range of metrics. 

Note that the output of this framework purely visual. While it is possible to assign quantifying scores to pick a 'best' model, we believe such techniques are reductionist, resulting in loss of important nuance. Reducing bias is usually a trade off with accuracy, and when it impacts real people the decisions about what boundaries are acceptable needs to be made by a person with domain knowledge, not a machine optimising with respect to some magic number we have invented. 

## Table of Contents

- [Bias Framework](#bias-framework)
- [Debiasing Graphs](#debiasing-graphs)
    - [Debiasing Graphs Object](#debiasing-graphs-object)
    - [Debiasing Graphs Composition](#debiasing-graphs-composition)
- [Metrics](#metrics)
- [Baselines](#baselines)
    - [Fairea Curve](#fairea-curve)
    - [baseline](#baseline)
- [Future](#future)


## Bias Framework
**Class**

The main class through which users will interact with this framework. Provides the interface to define the privileged group, apply debiasing techniques, and produce graphs. 

### \_\_init\_\_
Creates an instance of the bias framework applied to the specified model and data

Arguments:
``` 
model: The ML model to which the bias framework will be applied. This model must have a fit method and predict method. I am assuming at the moment that this will be an sklearn ml model, might try to modify this to be more flexible later.

df_x_train (pd.DataFrame): The training data features for the ML model.

df_x_validation (pd.DataFrame): The validation data features for the ML model.

df_y_train (pd.DataFrame): The training data labels for the ML model.

df_y_validation (pd.DataFrame): The validation data labels for the ML model.

pre_processing (optional): Any pre-processing to apply to the data before use by the ml model. Expected to implement methods for fit_transform and transform.
```

### set_privilege_function

Update the function which determines if an element belongs to the privileged or unprivileged class

Arguments:
```
privilege_function (callable): A function which when applied to a row of a dataframe will return a 1 for privileged and 0 for unprivileged. True and false values should also work.
```

### set_privileged_combinations

Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are considered privileged

Arguments
```
privileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored
            
Example from aif360 documentation:
[{'sex': 1, 'age': 1}, {'sex': 0}]
The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the privileged group
The second dictionary indicates that if sex has value 0 then the individual belongs to the privileged group (regardless of age)
```

### set_unprivileged_combinations

Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are not considered privileged

```
unprivileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored

Example from aif360 documentation:
[{'sex': 1, 'age': 1}, {'sex': 0}]
The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the unprivileged group
The second dictionary indicates that if sex has value 0 then the individual belongs to the unprivileged group (regardless of age)
```

### run_framework

Executes the framework using the model, data, and pre-processing provided at initialisation and the definition of privilege set using one of the provided methods. This will run through a number of debiasing methodologies and save the results. This may take some time. Once finished, you may either call one of the graph displaying methods of this class or the get_FaireaGraphsObject method to get an object which stores just the result and can be added to similar objects to combine the graphs.

### 