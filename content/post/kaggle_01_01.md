
---
author: Vidisha Jitani
date: 2018-07-26
linktitle: Awesomeness
title: My First Kaggle Competition - Feature Engineering Without Domain Knowledge
weight: 10
authorAvatar: hugo-logo.png
image: img/kaggle_3.png
published: true
---

This is the first article in the series of [machine learning](http://localhost:1313/post/my-experiments-with-kaggle---introduction/).
After a lot of brainstorming, I have finally decided the structure of this series. I'll pick up a challenge and will try to solve it in the best possible manner. I'll keep on posting all the new and awesome things that I would learn as I Kaggle further. 

Whenever any competition is announced, Kaggle sends an email notification for the same. So, always keep an eye on them. And this is how, I finally joined my first competition. I picked up [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge), primarily due to its small datasets. As I dont have an in-house GPU yet :( 

_How to get started??_ Go through the problem definition three to four times. And in case you are a novice like me, that might not be enough. So, checkout the discussion forums. Kaggle has an active community, so you might see almost all of your questions already being asked out there. Sometimes, even if you have a broader knowledge of how your logic will be, actually programming it can be very difficult, especially when its your first programme. And this is where almost 90% of normal engineers dropout. But, there are some good people still out their in the world, who give their solutions for free. And there is no shame in checking them out, and sometimes even understanding it is in itself a very difficult task. And yes, I was lucky enough to find an online solution.


So, lets get started with the challenge. This was a regression supervised learning problem. So, one should have at least some prerequisite knowledge about decision trees, random forest, boosting algorithms and neural nets. Also, this problem was specifically very difficult, because it didn't have any domain specifications. Also, the training set was toooo small as compared to the test dataset. So, even if your model will work fine on the training data, it might just crumpled down on the testing data. So, choosing the right features will play major role in the model's predictability accuracy, and thus feature engineering became the most important aspect of this particular problem. 

> Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive. 

But, what if you dont have any domain knowledge?? Can feature engineering be still attained??

**Feature Tranformation**

The target feature (which is being predicted) should have normal distribution. This is to ensure that the errors in the model have the same variance. That is, the error should not fluctuate with the values of the depending features. One should be able to say, _if I increase my variable X by 1, then on average and all else being equal, Y should increase by β1_. So, the target values distribution should be transformed to gaussian distributions. Usually, via inverse, logarithm or square roots this can be achieved. Since, all the values in the about target was greater than zero, logarithm tranformation was applied to normalize the data. 

```
   train = pd.read_csv("train.csv")
   y_train = train['target']
   y_train = np.log1p(y_train)
```


**Feature Removal**

If a feature is being taken into consideration in the model, it should provide some valuable insight which can help in prediction. So, there are some fairly easy way of removing some columns. If a column has same value for all the rows, drop them. If there are identical columns, drop the duplicates. 

```
   #Columns with all values same
   cols_with_onlyone_val = train.columns[train.nunique() == 1]
   train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
   
   #Remove identical columns
   NUM_OF_DECIMALS = 32
   train = train.round(NUM_OF_DECIMALS)
   colsToRemove = []
   columns = train.columns
   for i in range(len(columns)-1):
       v = train[columns[i]].values
       dupCols = []
       for j in range(i + 1,len(columns)):
           if np.array_equal(v, train[columns[j]].values):
               colsToRemove.append(columns[j])
   train.drop(colsToRemove, axis=1, inplace=True) 


```
A feature should consistently perform well on both training data and testing data. This can be ensured by checking out the distribution of the feature values. So, columns which doesn't have the same distribution in training and testing data, drop them. This was tested out via K-S test.

```
   from scipy.stats import ks_2samp
   THRESHOLD_P_VALUE = 0.01 #need tuned
   THRESHOLD_STATISTIC = 0.3 #need tuned
   diff_cols = []
   for col in train.columns:
       statistic, pvalue = ks_2samp(train[col].values, test[col].values)
       if pvalue <= THRESHOLD_P_VALUE and np.abs(statistic) > THRESHOLD_STATISTIC:
           diff_cols.append(col)
   for col in diff_cols:
       if col in train.columns:
           train.drop(col, axis=1, inplace=True)
           test.drop(col, axis=1, inplace=True)
   train.shape

```


**Feature Selection**

The dataset contained thousands of features. But, we wanted to choose only the most relevant features. But, why should we do this? _It makes the model simple, reduces overfitting and obviously the computational resources are preserved_. For selecting the most appropriate features, random forests algorithm is widely used. It is based on decision trees. These trees inherently rank all the features for choosing the splitting criteria of the tree. 

So, how does a decision tree works?? A Decision tree has a root node at the top. Based on an internal node/condition, a tree is split into two branches. One branch satisfies the condition, and the other one doesn't. The end of the branch that doesn't split anymore is the leaf node, which finally predicts the output. 
![0_Yclq0kqMAwCQcIV_.jpg](/img/0_Yclq0kqMAwCQcIV_.jpg)

So, on what criteria the condition is chosen? _The measure based on which the optimal condition is chosen is called impurity._ For classification problems, it is either Gini impurity or information gain and for regression tree problems, it is variance. So, the condition as per which the child nodes has the minimum impurity is chosen. So that ideally only one type of class is their in a node which will give 100% accuracy. The lesser the impurity in a node, means more accuracy and thus better prediction. Thus, all the features are ranked based on their ability to provide cleaner child nodes.


```
   from sklearn import model_selection
   from sklearn import ensemble
   
   NUM_OF_FEATURES = 1000
   def rmsle(y, pred):
       return np.sqrt(np.mean(np.power(y - pred, 2)))
   
   x1, x2, y1, y2 = model_selection.train_test_split(
       train, y_train.values, test_size=0.20, random_state=5)
   model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=7)
   model.fit(x1, y1)
   print(rmsle(y2, model.predict(x2)))
   
   col = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns}).sort_values(
       by=['importance'], ascending=[False])[:NUM_OF_FEATURES]['feature'].values
   train = train[col]
   train.shape

```


And thus via the above methods, we were able to considerably decrease the number of features and chose the most relevant features. With this, we were able to achieve score of 1.07 and a rank of 907/3736. And also with this submission, I became a contributor (One step closer :D)