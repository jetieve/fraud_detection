# Data science project for fraud detection

The data is about credit cards transactions, the goal of this project is to determine which transactions are fraudulent. The dataset has been PCA'd before it was made public, so there is no interpretability here. A more complete description of the dataset can be found here : https://www.kaggle.com/dalpozz/creditcardfraud

The predicted values are "fraudulent" or "normal", so it's a logistic regression problem. The main issue with this dataset is the huge imbalance in data : there are much more normal transactions than fraudulent ones, as can be seen here :

![Histogram](http://image.noelshack.com/fichiers/2017/40/3/1507074091-histogram.png)

There are 284315 normal transactions and 492 fraudulent transactions. We need to deal with this imbalance issue before proceeding with machine learning algorithms. For that, I split the data into 80% train set and 20% test set.

I used 2 resampling techniques to deal with this issue : 
- SMOTE : Synthetic Minority Over-sampling Technique, which is a combination of oversampling the minority class and undersampling the majority class (https://www.jair.org/media/953/live-953-2037-jair.pdf)
- ADASYN : Adaptive Synthetic Sampling Approach for Imbalanced Learning, which creates new artificial samples by focusing on data from the minority class for which it is harder to learn (http://140.123.102.14:8080/reportSys/file/paper/manto/manto_6_paper.pdf)

I chose to work with these algorithms :
- Neural network
- XGBoost
- Random Forest

Before training them on the SMOTE and ADASYN training sets, I extracted a subset of 20,000 individuals from the SMOTE dataset on which to train these algorithms with different sets of parameters through grid search, because there are many combinations to test and it would take too long with the full datasets. The neural network was the most complex to configure, because I had many parameters to test, including parameters for the neural network itself, and for the optimizer. All combinations would amount to over 3,000 models, so I had to train it different, each time considering only a subset of the parameters that I wanted to test, and keeping those values as constant for the next step, in order to reach an optimal solution.

The results showed me that I should use :
- a neural network with one hidden layer, 30 hidden units, a dropout of 5% and a relu activation function, and for the optimizer : a stochastic gradient descent, with a learning rate of 0.02, a momentum of 0.8 and using the Nesterov method
- for random forest : 1000 estimators, 8 features to consider for each split, not using bootstrap samples
- for XGBoost : a learning rate of 0.2, 1000 estimators, alpha regularization = 0, lambda regularization = 1

After I found those results, I applied these algorithms to the full SMOTE and ADASYN datasets to see which one of them would give me the best predictions :

![ROC Curve](http://image.noelshack.com/fichiers/2017/40/3/1507140281-roc-curve.png)

We can see here that the neural network applied to the SMOTE dataset (the green curve) is the best method, so we'll use this one to make our predictions. I tried to maximize the true negative rate while minimizing the false positive and false negative rates. I found that, starting from a probability threshold of 0.01, increasing the threshold up to 1 by an interval of 0.05 would only decrease the false positive rate, but had no impact on the true positive rate and the false negative rate.  In other words, it wouldn't help improving the classification for actual fradulent transactions, but it would decrease the amount of legit transactions classified as fraudulent transactions. This is the final confusion matrix : 

![Confusion matrix](http://image.noelshack.com/fichiers/2017/40/3/1507142532-confusion-matrix.png)

That means out of 107 fradulent transactions in the test set, 89 are classified correctly (83.1% good classification rate for this class), and 16 legit transactions are classified as fraudulent.

Next step : to improve detection of fradulents transactions, use a recall metric instead of an accuracy metric for XGBoost and neural network.
