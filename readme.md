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




# TODO 
I need to continue this documentation, currently at the graph interactions




## Future

Here I will list some features I would like to implement or have been suggested. 

### Selecting debiasing techniques 

Right now, the framework runs all debiasing techniques on the ml model. Some of these take quite a long time to run or a user may already have some knowledge of which debiasing techniques they would like to use. 

### Customisation of debiasing technique parameters 

There are parameters to the debiasing methodologies which can be modified. One would assume that for a given task some values are going to be better than others. Therefore, letting the user user their insight into the problem to optimise results could be a significant improvement. 

At the same time, we don't want to force the user to perform too must customisation, as this would increase the barrier to entry. 

### Debiasing methodologies to new file

Currently the debiasing methodologies are part of the framework class, we could reduce the complexity of this class and increase modularity by moving these to a new file in a similar form to metrics. Note, this would not really improve reusability since they rely on aif360.datasets

I would further like to split debiasing techniques into pre-processing and post-processing techniques, since there may be limitations regarding the usability of some techniques. For instance, pre-processing can make use of information not available when deployed, while post processing could act on existing models without re-training. 

In-processing should be considered, but unclear how.

### Greater customisability of graphs

I currently record a fair amount, but the user has no ability to use these in the graphs. For instance, say I would like to plot median with quartiles or customise how error is measured. 

### debiasing outcome object

I currently record results in dictionaries, however this provides a somewhat confusing experience and makes documentation more difficult as I need to explain the structure of the dictionary. Creating an object to handle this interaction could increase or decrease clarity. This requires further thought.

### Simplify framework class

I feel like this framework is either already complex enough or will become complex enough that the main class should do nothing more than orchestrate the interaction between classes. 

### Random seed for bias framework

I have already implemented random seeds for some functions, but not everywhere. Right now, there is no way for a user to enforce a consistent result, which is important for testing. 

### Unit test code

The methods producing graphs probably aren't unit testable beyond checking that they do not raise exceptions, but there still exists some functionality which can be tested.

### More customisable bias

I would like to debias with respect to one definition of privilege and be able to measure the impact on another definition to better understand complex relationships