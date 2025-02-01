# Foundations of Machine Learning
📌 Understand the basics of Machine Learning
##### Source: [NVIDIA Machine Learning](https://www.nvidia.com/en-us/glossary/machine-learning/)
### Machine Learning
Machine learning (ML) is a two-phase process where computer systems find patterns in massive amounts of data and make predictions based on the patterns. For each phase, a model is used in the form of algorithms or statistical concepts. In simple terms, machine learning trains a machine to learn without it being programmed to do so.

Machine learning uses algorithms to autonomously create models from data fed into a machine learning platform. Typical programmed systems rely on expertise knowledge in programmed rules, however, when data is changing, the rules are difficult to maintain. **ML provides the ability to learn from increasing volumes of data and provide data driven predictions.**

The performance of an ML algorithm depends on the capability of algorithms turning a dataset into a model. Depending on the task, different algorithms are needed. Additionally, performing the task relies on the quality of the input data to the model.

Machine learning implements two techniques that divide the use of algorithms into different types: **supervised** and **un-supervised** learning.
### Supervised Learning
Supervised learning is an analysis where an algorithms are used to train models to find patterns in data with **labels and features**. It uses the trained model to predict the labels on new datasets.

![Supervised Learning](Resources/supervised_learning.png)

There are two flavours of supervised learning; **classification** and **regression**.

#### Classification
Classification identifies the category that an item belongs to based on labeled examlples of known items.

An example of classification is the following scenario.

**Example**: Fraudelent Credit Card Transactions

**Label**: Probability of Fraud

**Features**:
1. IBAN
2. Transaction amount
3. Location

![Classification](Resources/classification.png)

Logistic regression can be used to estimate the probabilty whether a transaction is fraudelent (label) depending on the IBAN, amount, and location information (features).

#### Regression
Regression estimates the relationship between a target outcome label and one or more feature variables to predict a continuous numeric value.

An example of regression is the following scenario.

**Example**: House Prices

**Label**: Price of Houses

**Features**:
1. Size

![Regression](Resources/regression.png)

Linear regression can be used to predict the house price (label) based on the size of the house (feature).

### Un-supervised Learning
TBA

### Benefits of Machine Learning
TBA

### Machine Learning Use Cases
TBA

### Why it matters?
TBA

### Benefits of using GPUs
TBA

