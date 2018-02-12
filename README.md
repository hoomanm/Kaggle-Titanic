# Kaggle's Titanic Competition: Machine Learning from Disaster

The aim of this project is to predict which passengers survived the Titanic tragedy given a set of labeled data as the training dataset. Our strategy is to identify an informative set of features and then try different classification techniques to attain a good accuracy in predicting the class labels. Our results indicate that ensemble methods perform better than other algorithms. Specifically, considering the majority vote of a number of classifiers including the ensemble models obtains the best accuracy.

We are given a training dataset with 891 samples and a test dataset with 418 samples including 9 features. After feature engineering, which involves removing some of the features and creating a couple of new ones by combining other features, we keep 10 features. These features are the most important ones according to the feature importnaces produced by different trained models.

Once we have our set of features, in order to examine different classifiers, we use 5-fold cross-validation technique on the training dataset. in addition to the cross-validation, we compute the accuracy of the models using the test dataset provided by Kaggle. The below table shows the cross-validation scores and the scores achieved on the Kaggle leaderboard for our different models:

| Model | Accuracy (5-fold Cross Validation) | Accuracy (Kaggle Test Dataset) |
| ----- | ---------------------------------- | ------------------------------ |
| Voting Classifier | 0.8272 | 0.8086 |
| Two-Layer XGBoost | 0.8431 | 0.7847 |
| Random Forest | 0.8070 | 0.7751 |
| XGBoost | 0.8194 | 0.7703 |
| Support Vector Classifier | 0.7867 | 0.7655 |
| Decision Tree | 0.7185 | 0.6651 |

It is worth mentioning that in order to train the two-layer XGBoost model, first we train a couple of other classifiers such as: AdaBoost, Gradient Boosting, Random Forest, and Extra Trees. Then, we use the output of these classifiers as the input features for the XGBoost classifier. 

The Voting Classifier uses two-layer XGBoost, Logistic Regression, K-Nearest Neighbor, Support Vector Classifier, and Random Forest all together and then predicts the class label that has the majority of the votes. We also use different weights for different classifiers based on their influence on the accuracy.
