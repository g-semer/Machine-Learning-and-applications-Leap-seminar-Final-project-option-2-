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
We load the data using `fetch_opeth` as a `X dataframe` and a `y target list`. Then we preview a part of the dataset using `X.head()`, we check its shape with `X.shape` and count how many higher and lower income households the dataset includes. In the end we get:
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

## Full pipeline
To create the full pipeline needed for the logistic regression model we combined in a pipeline the `preprocess` and `LogisticRegression` for model, with `max_iter=1000` so that the model converges.

## Cross-validation
After that we estimate the model's performance with cross-validation on the training data using `StratifiedKFold`, averaging over 5 folds to get a steadier estimate.  
Then we compute the cross-validation scores with `cross_val_score`, scoring by F1 and print the mean and the standard deviation:
```
Mean ± Std (Train set):  0.6576 ± 0.0098
```

## Fitting and evaluating the test set
In the end we fit the training data on the `clf` pipeline, make a prediction for the `X_test` data using the `clf` pipeline, and evaluate the accuracy score, the F1 score and the confusion matrix by comparing the `y_test` and `y_pred_clf`, as shown below:
```
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)

clf_acc = accuracy_score(y_test, y_pred_clf)
clf_f1 = f1_score(y_test, y_pred_clf)
confusion_matrix_clf = confusion_matrix(y_test, y_pred_clf)

print("Accuracy: ", np.round(clf_acc, 4))
print("F1 score: ", np.round(clf_f1, 4))
print()
print("Confusion Matrix (Logistic Regression):")
print(confusion_matrix_clf)
```
```
Accuracy:  0.8524
F1 score:  0.6562
Confusion Matrix (Logistic Regression):
[[6951  480]
 [ 962 1376]]
```


# Neural Network

## Fitting and transforming the data
A neural network needs a plain numeric array as input, so we apply the `preprocess` to turn the mixed table into numbers. We fit the `preprocess` on the training dataframes only, while transform both sets, using `fit_transform` for `X_train` and just `transform` for `X_test`.

## Building the network
After getting the data ready, we build a small dense network using `keras.Sequential` with the following layers:
* An `Input` layer with as many units as the transformed train dataframe.
* 2 hidden `Dense` layers with activation function `relu`, giving 64 units for the first and 32 for the second.
* 2 `Dropout` layers with 0.3 rate, randomly deactivating 30% of the above layer's units to reduce overfitting.
* One output `Dense` layer with activation function `sigmoid` and 1 unit, because the target is binary, to give the probability of the positive (higher income) class.
The networks layout is as shown below:
```
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_9 (Dense)                 │ (None, 64)             │         6,784 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_6 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_10 (Dense)                │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_7 (Dropout)             │ (None, 32)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_11 (Dense)                │ (None, 1)              │            33 │
```

## Compiling and training the network
We compile the network using `adam` as the optimizer, `accuracy` for the metrics and `binary_crossentropy` as the loss function because the output is a single sigmoid unit.\
\
Then we create a `callback` variable with `EarlyStopping`, monitoring `val_loss`, giving it a patience of `3 epochs` before checking and using `restore_best_weight`, to keep the best weights before the validation loss stopped improving.\
\
Now we can fit the network to training data, using the `callback`, `20 epochs`, `batch_size 128` and `validation_split 0.1` to evaluate the model during it's training.

## Evaluating on the test set
Continuing we make the prediction on the transformed test set. The network outputs probabilities, so we create a `nn_pred` list for the prediction whose values are positive when the probability is above 0.5, as shown below:
```
nn_prob = net.predict(X_test_prep).ravel()
nn_pred = (nn_prob > 0.5).astype(int)
```
In the end we evaluate the accuracy score, the F1 score and the confusion matrix by comparing the `y_test` and `nn_pred`, as shown below:
```
Accuracy:  0.8552
F1 score:  0.6643
Confusion Matrix (Neural Network):
[[6954  477]
 [ 938 1400]]
```


# The Comparison







