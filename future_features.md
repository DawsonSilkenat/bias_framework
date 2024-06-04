## Future

Here I will list some features I would like to implement or have been suggested. Some of these will be user facing, others are more focused on maintainability.

### Selecting debiasing techniques 

Right now, the framework runs all debiasing techniques on the ml model. Some of these take quite a long time to run or a user may already have some knowledge of which debiasing techniques they would like to use. 

### Customisation of debiasing technique parameters 

There are parameters to the debiasing methodologies which can be modified. One would assume that for a given task some values are going to be better than others. Therefore, letting the user user their insight into the problem to optimise results could be a significant improvement. 

At the same time, we don't want to force the user to perform too must customisation, as this would increase the barrier to entry. 


### debiasing outcome object

I currently record results in dictionaries, however this provides a somewhat confusing experience and makes documentation more difficult as I need to explain the structure of the dictionary. Creating an object to handle this interaction could increase or decrease clarity. This requires further thought.

### Simplify framework class

I feel like this framework is either already complex enough or will become complex enough that the main class should do nothing more than orchestrate the interaction between classes. 


### Unit test code

The methods producing graphs probably aren't unit testable beyond checking that they do not raise exceptions, but there still exists some functionality which can be tested.

### More customisable bias

I would like to debias with respect to one definition of privilege and be able to measure the impact on another definition to better understand complex relationships

### DebiasingGraphsComposition inheritance 

I would like DebiasingGraphsComposition to inherit from DebiasingGraphsObject to make type checking easier, however it then inherits all the functions so \_\_getattr\_\_ is not called, breaking functionality.

### Input validation check

If either our privileged or unprivileged groups lacks instances of the positive or negative class we will encounter issues, and speaks to a strong bias in the data. It may be worth checking for such a situation before running the framework.