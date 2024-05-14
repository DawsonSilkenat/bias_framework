**APPData.ipynb**

The APP Fraud Synthetic Data dataset contains the results of agent-based simulations. We are using it for the personal_customers, personal_transactions, and people tables. We will use these estimate relationships between characteristics of an individual and their spending habits.

From these datasets we drop some details which would be unreasonable to include for classification purposes, for instance sort code, account number, and name.

This dataset was designed with fraud in mind, and as such has a larger than expected number of fraudulent interactions. We remove all entries related to fraud so that we end up with a dataset more representative of typical individuals and transactions.

Since the APP data includes records of transactions and the transaction type, we can calculate a number of features, for instance the income of an individual or summary statistics on their spending habits.

We can do something similar with their account balance. We can also create some boolean classifications based on the existence of certain transaction types, namely receiving benefits, receiving pension, receiving salary, has mortgage, and internet access

In total, this gives us a dataset containing the details of an individual and their spending habits.One piece of information contained in our dataset is a postcode, which we can use to join with the index of multiple deprivation. 

To summarise, using the APP datasets we have produced a dataset that contains individuals personal information, transaction information, and IMD information 


**WEETSData.ipynb**

We are using the WEETS Reference - Synthetic Individuals dataset to estimate relationships between characteristics of an individual and their credit score, which we use as an estimate for their financial inclusion. 

The WEETS data processing is pretty straightforward: we simple need to convert some columns to a more useable format, for instance converting a passport number to a boolean indicating the existence of an international passport, then drop a few unneeded columns. Since the WEETS dataset also includes a postcode, we can again join to the index of multiple deprivation. This results in a dataset containing personal information, IMD data, and credit scores


**Joining_WEETS_APP.ipynb**

Once we have ran the two files mentioned above we have two datasets containing information about individuals. One additionally have information on transactions while the other has information about credit score. Here we seek to combine these into one dataset containing both pieces of information. In particular, for each entity in the WEETS data we will find a distribution for the spending habits of similar individuals in the APP data, from which we will generate a random number to use for the WEETS individual. 

Before performing a join, we must select which APP spending features we would like to add to the WEETS data. There are three possible joins: common characteristics, index of multiple deprivation properties, or geographical location.


To join based on common characteristics, we must select which boolean columns from among the common features we would like to partition the datasets on. We will also group by age, where anyone below the age of 20 has been removed, and split into buckets 10 years (20 to 30, 30 to 40, etc).  This partitioning is first applied to the APP data. We can then iterate over the partition and calculate a normal distribution for each of the columns we would like to include from the app data into the WEETS data. This is then merged into the WEETS data and a random value from each of the generated distributions is drawn

To join based on the index of multiple deprivation we follow a very similar format to what we did with age in the previous join.We first cover the IMD values to bins, then partitioning the APP dataset using these value. We calculate a normal distribution for the values in each partition, then merge into the WEET data and draw a random value from the generated distributions.

The geographic join is even simpler. For each individual in the WEETS dataset, we find the 5 individuals in the APP dataset which are geographically closest and take the average of their spending habits. Note that this is the only join that does not involve sampling from a distribution. 