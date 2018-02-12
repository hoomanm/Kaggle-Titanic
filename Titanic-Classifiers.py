import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb


train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
combine = [train_df, test_df]
original_test_df = test_df

######################### Feature Engineering #########################
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
	                                         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 2, 'C': 1, 'Q': 0} ).astype(int)

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

for dataset in combine:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['FamilySize'] + 1)

for dataset in combine:
    dataset['IsAlone'] = 1
    dataset.loc[dataset['FamilySize'] == 0, 'IsAlone'] = 0

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

for dataset in combine:
    dataset.loc[ dataset['Fare_Per_Person'] <= (7.91 / (dataset['FamilySize'] + 1)), 'Fare_Per_Person'] = 0
    dataset.loc[(dataset['Fare_Per_Person'] > (7.91 / (dataset['FamilySize'] + 1))) & (dataset['Fare_Per_Person'] <= (14.454 / (dataset['FamilySize'] + 1))), 'Fare_Per_Person'] = 1
    dataset.loc[(dataset['Fare_Per_Person'] > (14.454 / (dataset['FamilySize'] + 1))) & (dataset['Fare_Per_Person'] <= (31 / (dataset['FamilySize'] + 1))), 'Fare_Per_Person']   = 2
    dataset.loc[ dataset['Fare_Per_Person'] > (31 / (dataset['FamilySize'] + 1)), 'Fare_Per_Person'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

##### Gives the length of the name #####
train_df['Name_length'] = train_df['Name'].apply(len)
test_df['Name_length'] = test_df['Name'].apply(len)

##### Feature that tells whether a passenger had a cabin on the Titanic #####
train_df['Has_Cabin'] = train_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_df['Has_Cabin'] = test_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


############## Normalization ##################
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_df_cols = train_df.columns
feature_cols = train_df_cols.drop("Survived")

min_max_scaler = preprocessing.MinMaxScaler()
train_df[feature_cols] = min_max_scaler.fit_transform(train_df[feature_cols])

test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
test_df[feature_cols] = min_max_scaler.fit_transform(test_df[feature_cols])

X_train = train_df[feature_cols]
Y_train = train_df['Survived']
X_test = test_df[feature_cols]


################### Some useful parameters ###################
ntrain = train_df.shape[0]
ntest = test_df.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits= NFOLDS, random_state=SEED)

######## Geting Out-of-Fold predictions ########
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

######## Class to extend the Sklearn classifier ########
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        print str(self.clf).split('(')[0], ":", self.clf.fit(x,y).feature_importances_

############ Parameters for the classifiers ############
#### Random Forest parameters ####
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

#### Extra Trees Parameters ####
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

#### AdaBoost parameters ####
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

#### Gradient Boosting parameters ####
gb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

#### Support Vector Classifier parameters ####
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

#### K-Nearest Neighbors parameters ####
nn_params = {
    'n_neighbors': 3
}

########### Creating the models ###########
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

###### Creating Numpy arrays of train, test and target (Survived) dataframes ######
y_train = train_df['Survived'].values
x_train = train_df[feature_cols].values
x_test = test_df[feature_cols].values


###### Creating our OOF train and test predictions to be used as new features ######
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test) # Support Vector Classifier


################ Feature Importances ####################
print "Feature Importances:"
rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


########## New Features ###############
base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
      'SVC': svc_oof_train.ravel()
    })
# print base_predictions_train.head()
# print base_predictions_train.describe()

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


###### XGBoost as the Second-level Classifier using the predictions of above classifiers as input features ######
gbm = xgb.XGBClassifier(
     n_estimators= 1000,
     max_depth= 4,
     min_child_weight= 2,
     gamma=0.9,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread= -1,
     scale_pos_weight=1)

scores = cross_val_score(gbm, x_train, y_train, cv=5)
print "XGBoost CV Score: ", scores.mean()


#######********** Other Classifiers using the original set of features **********#######
######## Logistic Regression ########
logreg = LogisticRegression()
scores = cross_val_score(logreg, X_train, Y_train, cv=5)
print "Logistic Regression CV Score: ", scores.mean()


######## Support Vector Classifier ########
svc = SVC(kernel= 'linear', C= 0.025)
scores = cross_val_score(svc, X_train, Y_train, cv=5)
print "SVM CV Score: ", scores.mean()

######## Perceptron ########
perceptron = Perceptron()
scores = cross_val_score(perceptron, X_train, Y_train, cv=5)
print "Perceptron CV Score: ", scores.mean()

######## Stochastic Gradient Descent ########
sgd = SGDClassifier()
scores = cross_val_score(sgd, X_train, Y_train, cv=5)
print "Stochastic GD CV Score: ", scores.mean()

######## K-Nearest Neighbors ########
knn = KNeighborsClassifier(n_neighbors = 3)
scores = cross_val_score(knn, X_train, Y_train, cv=5)
print "KNN CV Score: ", scores.mean()

######## Decision Tree ########
decision_tree = DecisionTreeClassifier()
scores = cross_val_score(decision_tree, X_train, Y_train, cv=5)
print "Decision Tree CV Score: ", scores.mean()

######## Random Forest ########
random_forest = RandomForestClassifier(n_estimators=500)
scores = cross_val_score(random_forest, X_train, Y_train, cv=5)
print "Random Forest CV Score: ", scores.mean()

######## AdaBoost ########
ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.75)
scores = cross_val_score(ada, X_train, Y_train, cv=5)
print "AdaBoost CV Score: ", scores.mean()

######## Extra Trees ########
et = ExtraTreesClassifier(n_jobs=-1,
n_estimators=500,
max_depth=8,
min_samples_leaf=2,
verbose=0)
scores = cross_val_score(et, X_train, Y_train, cv=5)
print "ExtraTrees CV Score: ", scores.mean()

######## Gradient Boosting ########
gb = GradientBoostingClassifier(n_estimators=500,
max_depth=5,
min_samples_leaf=2,
verbose=0)
scores = cross_val_score(gb, X_train, Y_train, cv=5)
print "GradientBoosting CV Score: ", scores.mean()

######## Voting Classifier ########
clf_vc = VotingClassifier(estimators=[('xgb1', gbm), ('lg1', logreg), ('svc', svc),
                                      ('rfc1', random_forest), ('knn', knn)],
                          voting='hard', weights=[3,1,1,2,1])

# clf_vc = VotingClassifier(estimators=[('xgb1', gbm), ('ada', ada), ('et', et), ('gb', gb), ('lg1', logreg), ('svc', svc),
#                                       ('rfc1', random_forest), ('knn', knn), ('dc', decision_tree)],
#                           voting='hard', weights=[2,2,2,2,1,1,2,1,1])

clf_vc = clf_vc.fit(X_train, Y_train)
Y_pred_voting = clf_vc.predict(X_test)
scores = cross_val_score(clf_vc, X_train, Y_train, cv=5)
print "Voting Classifier CV Score: ", scores.mean()


################ Preparing the results for submission (using voting classifier) ################
submission = pd.DataFrame({
        "PassengerId": original_test_df["PassengerId"],
        "Survived": Y_pred_voting.astype(int)
    })
submission.to_csv('./submission.csv', index=False)




