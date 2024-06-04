# TODO I'm considering storing the results of debiasing as an object rather than a dictionary. Need to consider what advantage this would offer


""" 
Pros:
    - Functions for accessing data allow for 
        - autocomplete
        - default values
        - immutability of internal data
    - Could offer functions to access the data in more formats easily, for instance pd.Dataframe
    - Allows easier modification without wide reaching code modifications
    - Possibly more readable access to data

Cons: 
    - Additional class to be maintained and documented 
    - Unfamiliar to users, unlike dict or pd.Dataframe which I could expect most users to know how to explore
"""


class DebiasOutcome:
    pass