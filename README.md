# Machine-Learning-and-applications-Leap-seminar-Final-project-option-2
### (Option 2: Build a classical model and a neural network for the same data set, then compare them fairly)
In the final project we had built one logistic regression model and one neural network for the same dataset and compare which of them makes the best predictions as well as which is better for the scenario given.

## The scenario
A public agency wants to flag likely higher-income households (more than 50000 dollars a year) from survey data, so that an outreach programme can be targeted, and it must choose between a simple, explainable model and a more complex neural network. In this project you build both and judge whether the added complexity is worthwhile.

## The dataset
The dataset used is Adult, also called Census Income. Each row describes one person, with a mix of numeric features (such as age and hours worked per week) and categorical features (such as occupation and education). The task is to predict whether that person earns more than 50,000 dollars a year.

## Libraries used
- **Scikit-Learn:** Dataset 'adult', Splitting the dataset, Preprocessing, Creating the logistic regression model, Evaluating metrics.
- **Keras:** Sequential API for building and training the neural network.
- **Pandas:**
- **Numpy:** 
- **Matplotlib and Seaborn:** Metrics visualization.


# Processing the data

## Loading and examining the data
We load the data using `fetch_opeth` as a X dataframe and a y list. Then we preview a part of the dataset using `X.head()`, we check its shape with `X.shape` and count how many higher and lower income households the dataset includes. In the end we get:
```
Rows and columns: (48842, 14)
Target values: {'<=50K': 37155, '>50K': 11687}
```
As we can see there is a big difference between the number of higher and lower income households with the higher income being the minority.  

## Splitting the data
After loading and examining the data we split the in two sets:
- A train set that we will use only to train the model with.
- A test set that the model will try to predict the targets y while only able to see the values X.
By splitting the data in train and test sets we make sure that the model's prediction isn't influenced by the the target values.
Then we split the train set once again in numeric and categorical features because neither of the models is able to understand text (categorical) values and so we will have to preprocess each type differently.

## Preprocessing
### Numeric pipeline
We make a pipeline for the numeric features using `SimpleImputer` to fill missing values and `StandardScaler` to scale the values around zero.
### Categorical pipeline
We make a pipeline for the numeric features using `SimpleImputer` to fill missing values and `OneHotEncoder` to turn categories into numbers.\
\
\
Then we combine the two pipelines in variable called `preprocess` using `ColumnTransformer`, applying each pipeline to its own list of columns (numeric features and categorical features).


# Logistic Regression Model

##

