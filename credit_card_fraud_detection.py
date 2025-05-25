#!/usr/bin/env python
# coding: utf-8

# ### Problem statement:-
# 
# The aim of the project is to predict fraudulent credit card transactions using machine learning models. This is crucial from the bank’s as well as customer’s perspective. The banks cannot afford to lose their customers’ money to fraudsters. Every fraud is a loss to the bank as the bank is responsible for the fraud transactions.
# 
# The dataset contains transactions made over a period of two days in September 2013 by European credit cardholders. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. We need to take care of the data imbalance while building the model and come up with the best model by trying various algorithms. 
# 

# ## Steps:-
# The steps are broadly divided into below steps. The sub steps are also listed while we approach each of the steps.
# 1. Reading, understanding and visualising the data
# 2. Preparing the data for modelling
# 3. Building the model
# 4. Evaluate the model

# In[2]:


# This was used while running the model in Google Colab
# from google.colab import drive
# drive.mount('/content/drive')


# Importing the libraries
import pandas as pd
import numpy as np

import matplotlib as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns', 500)


# # Exploratory data analysis

# ## Reading and understanding the data

# In[3]:


# Reading the dataset
df = pd.read_csv('creditcard.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# ## Handling missing values

# #### Handling missing values in columns

# In[7]:


# Cheking percent of missing values in columns
df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# We can see that there is no missing values in any of the columns. Hence, there is no problem with null values in the entire dataset.

# ### Checking the distribution of the classes

# In[8]:


classes = df['Class'].value_counts()
classes


# In[9]:


normal_share = round((classes[0]/df['Class'].count()*100),2)
normal_share


# In[10]:


fraud_share = round((classes[1]/df['Class'].count()*100),2)
fraud_share


# We can see that there is only 0.17% frauds. We will take care of the class imbalance later.

# In[11]:


# Bar plot for the number of fraudulent vs non-fraudulent transcations
sns.countplot(x='Class', data=df)
plt.title('Number of fraudulent vs non-fraudulent transcations')
plt.show()


# In[12]:


# Bar plot for the percentage of fraudulent vs non-fraudulent transcations
fraud_percentage = {'Class':['Non-Fraudulent', 'Fraudulent'], 'Percentage':[normal_share, fraud_share]} 
df_fraud_percentage = pd.DataFrame(fraud_percentage) 
sns.barplot(x='Class',y='Percentage', data=df_fraud_percentage)
plt.title('Percentage of fraudulent vs non-fraudulent transcations')
plt.show()


# ## Outliers treatment

# We are not performing any outliers treatment for this particular dataset. Because all the columns are already PCA transformed, which assumed that the outlier values are taken care while transforming the data.

# ### Observe the distribution of classes with time

# In[13]:


# Creating fraudulent dataframe
data_fraud = df[df['Class'] == 1]
# Creating non fraudulent dataframe
data_non_fraud = df[df['Class'] == 0]


# In[14]:


# Distribution plot
plt.figure(figsize=(8,5))
ax = sns.distplot(data_fraud['Time'],label='fraudulent',hist=False)
ax = sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
ax.set(xlabel='Seconds elapsed between the transction and the first transction')
plt.show()


# ##### Analysis
# We do not see any specific pattern for the fraudulent and non-fraudulent transctions with respect to Time.
# Hence, we can drop the `Time` column.

# In[15]:


# Dropping the Time column
df.drop('Time', axis=1, inplace=True)


# ### Observe the distribution of classes with amount

# In[16]:


# Distribution plot
plt.figure(figsize=(8,5))
ax = sns.distplot(data_fraud['Amount'],label='fraudulent',hist=False)
ax = sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
ax.set(xlabel='Transction Amount')
plt.show()


# ##### Analysis
# We can see that the fraudulent transctions are mostly densed in the lower range of amount, whereas the non-fraudulent transctions are spreaded throughout low to high range of amount. 

# ## Train-Test Split

# In[19]:


# Import library
from sklearn.model_selection import train_test_split


# In[20]:


# Putting feature variables into X
X = df.drop(['Class'], axis=1)


# In[21]:


# Putting target variable to y
y = df['Class']


# In[22]:


# Splitting data into train and test set 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)


# ## Feature Scaling
# We need to scale only the `Amount` column as all other columns are already scaled by the PCA transformation.

# In[23]:


# Standardization method
from sklearn.preprocessing import StandardScaler


# In[24]:


# Instantiate the Scaler
scaler = StandardScaler()


# In[25]:


# Fit the data into scaler and transform
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])


# In[26]:


X_train.head()


# ##### Scaling the test set
# We don't fit scaler on the test set. We only transform the test set.

# In[27]:


# Transform the test set
X_test['Amount'] = scaler.transform(X_test[['Amount']])
X_test.head()


# ## Checking the Skewness

# In[28]:


# Listing the columns
cols = X_train.columns
cols


# In[29]:


# Plotting the distribution of the variables (skewness) of all the columns
k=0
plt.figure(figsize=(17,28))
for col in cols :    
    k=k+1
    plt.subplot(6, 5,k)    
    sns.distplot(X_train[col])
    plt.title(col+' '+str(X_train[col].skew()))


# We see that there are many variables, which are heavily skewed. We will mitigate the skewness only for those variables for bringing them into normal distribution.

# ### Mitigate skweness with PowerTransformer

# In[30]:


# Importing PowerTransformer
from sklearn.preprocessing import PowerTransformer
# Instantiate the powertransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
# Fit and transform the PT on training data
X_train[cols] = pt.fit_transform(X_train)


# In[32]:


# Transform the test set
X_test[cols] = pt.transform(X_test)


# In[34]:


# Plotting the distribution of the variables (skewness) of all the columns
k=0
plt.figure(figsize=(17,28))
for col in cols :    
    k=k+1
    plt.subplot(6, 5,k)    
    sns.distplot(X_train[col])
    plt.title(col+' '+str(X_train[col].skew()))


# Now we can see that all the variables are normally distributed after the transformation.

# # Model building on imbalanced data

# ### Metric selection for heavily imbalanced data
# As we have seen that the data is heavily imbalanced, where only 0.17% transctions are fraudulent, we should not consider Accuracy as a good measure for evaluating the model. Because in the case of all the datapoints return a particular class(1/0) irrespective of any prediction, still the model will result more than 99% Accuracy.
# 
# Hence, we have to measure the ROC-AUC score for fair evaluation of the model. The ROC curve is used to understand the strength of the model by evaluating the performance of the model at all the classification thresholds. The default threshold of 0.5 is not always the ideal threshold to find the best classification label of the test point. Because the ROC curve is measured at all thresholds, the best threshold would be one at which the TPR is high and FPR is low, i.e., misclassifications are low. After determining the optimal threshold, we can calculate the F1 score of the classifier to measure the precision and recall at the selected threshold.

# #### Why SVM was not tried for model building and Random Forest was not tried for few cases?
# In the dataset we have 284807 datapoints and in the case of Oversampling we would have even more number of datapoints. SVM is not very efficient with large number of datapoints beacuse it takes lot of computational power and resources to make the transformation. When we perform the cross validation with K-Fold for hyperparameter tuning, it takes lot of computational resources and it is very time consuming. Hence, because of the unavailablity of the required resources and time SVM was not tried.
# 
# For the same reason Random forest was also not tried for model building in few of the hyperparameter tuning for oversampling technique.

# #### Why KNN was not used for model building?
# KNN is not memory efficient. It becomes very slow as the number of datapoints increases as the model needs to store all the data points. It is computationally heavy because for a single datapoint the algorithm has to calculate the distance of all the datapoints and find the nearest neighbors.

# ### Logistic regression

# In[35]:


# Importing scikit logistic regression module
from sklearn.linear_model import LogisticRegression


# In[36]:


# Impoting metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# #### Tuning hyperparameter  C
# C is the the inverse of regularization strength in Logistic Regression. Higher values of C correspond to less regularization.

# In[37]:


# Importing libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[39]:


# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as recall as we are more focused on acheiving the higher sensitivity than the accuracy
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train, y_train)


# In[41]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[42]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('roc_auc')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')


# In[43]:


# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test roc_auc is {0} at C = {1}".format(best_score, best_C))


# #### Logistic regression with optimal C

# In[44]:


# Instantiate the model with best C
logistic_imb = LogisticRegression(C=0.01)


# In[45]:


# Fit the model on the train set
logistic_imb_model = logistic_imb.fit(X_train, y_train)


# ##### Prediction on the train set

# In[46]:


# Predictions on the train set
y_train_pred = logistic_imb_model.predict(X_train)


# In[47]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[48]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[49]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train, y_train_pred))


# In[50]:


# classification_report
print(classification_report(y_train, y_train_pred))


# ##### ROC on the train set

# In[51]:


# ROC Curve function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[52]:


# Predicted probability
y_train_pred_proba = logistic_imb_model.predict_proba(X_train)[:,1]


# In[53]:


# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba)


# We acheived very good ROC 0.99 on the train set.

# #### Prediction on the test set

# In[54]:


# Prediction on the test set
y_test_pred = logistic_imb_model.predict(X_test)


# In[55]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[56]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[57]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_test, y_test_pred))


# In[58]:


# classification_report
print(classification_report(y_test, y_test_pred))


# ##### ROC on the test set

# In[59]:


# Predicted probability
y_test_pred_proba = logistic_imb_model.predict_proba(X_test)[:,1]


# In[60]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# We can see that we have very good ROC on the test set 0.97, which is almost close to 1.

# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 0.70
#     - Specificity = 0.99
#     - F1-Score = 0.76
#     - ROC = 0.99
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.77
#     - Specificity = 0.99
#     - F1-Score = 0.65
#     - ROC = 0.97
# 
# Overall, the model is performing well in the test set, what it had learnt from the train set.

# ### XGBoost

# In[62]:


# Importing XGBoost
from xgboost import XGBClassifier


# ##### Tuning the hyperparameters

# In[63]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)       


# In[64]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[65]:


# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):


    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# ##### Model with optimal hyperparameters
# We see that the train score almost touches to 1. Among the hyperparameters, we can choose the best parameters as learning_rate : 0.2 and subsample: 0.3

# In[66]:


model_cv.best_params_


# In[67]:


# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for calculating auc
params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
xgb_imb_model = XGBClassifier(params = params)
xgb_imb_model.fit(X_train, y_train)


# ##### Prediction on the train set

# In[68]:


# Predictions on the train set
y_train_pred = xgb_imb_model.predict(X_train)


# In[69]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[70]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[71]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train, y_train_pred))


# In[72]:


# classification_report
print(classification_report(y_train, y_train_pred))


# In[73]:


# Predicted probability
y_train_pred_proba_imb_xgb = xgb_imb_model.predict_proba(X_train)[:,1]


# In[74]:


# roc_auc
auc = metrics.roc_auc_score(y_train, y_train_pred_proba_imb_xgb)
auc


# In[75]:


# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba_imb_xgb)


# ##### Prediction on the test set

# In[76]:


# Predictions on the test set
y_test_pred = xgb_imb_model.predict(X_test)


# In[77]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[78]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[79]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_test, y_test_pred))


# In[80]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[81]:


# Predicted probability
y_test_pred_proba = xgb_imb_model.predict_proba(X_test)[:,1]


# In[82]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[83]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 0.85
#     - Specificity = 0.99
#     - ROC-AUC = 0.99
#     - F1-Score = 0.90
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.75
#     - Specificity = 0.99
#     - ROC-AUC = 0.98
#     - F-Score = 0.79
# 
# Overall, the model is performing well in the test set, what it had learnt from the train set.

# ### Decision Tree

# In[84]:


# Importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier


# In[85]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 3, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)


# In[89]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[90]:


# Printing the optimal sensitivity score and hyperparameters
print("Best roc_auc:-", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[91]:


# Model with optimal hyperparameters
dt_imb_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=100,
                                  min_samples_split=100)

dt_imb_model.fit(X_train, y_train)


# ##### Prediction on the train set

# In[92]:


# Predictions on the train set
y_train_pred = dt_imb_model.predict(X_train)


# In[93]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train)
print(confusion)


# In[94]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[95]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train, y_train_pred))


# In[96]:


# classification_report
print(classification_report(y_train, y_train_pred))


# In[97]:


# Predicted probability
y_train_pred_proba = dt_imb_model.predict_proba(X_train)[:,1]


# In[98]:


# roc_auc
auc = metrics.roc_auc_score(y_train, y_train_pred_proba)
auc


# In[99]:


# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba)


# ##### Prediction on the test set

# In[100]:


# Predictions on the test set
y_test_pred = dt_imb_model.predict(X_test)


# In[101]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[102]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[103]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train, y_train_pred))


# In[104]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[105]:


# Predicted probability
y_test_pred_proba = dt_imb_model.predict_proba(X_test)[:,1]


# In[106]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[107]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 1.0
#     - Specificity = 1.0
#     - F1-Score = 0.75
#     - ROC-AUC = 0.95
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.58
#     - Specificity = 0.99
#     - F-1 Score = 0.75
#     - ROC-AUC = 0.92
# 

# ### Random forest

# In[108]:


# Importing random forest classifier
from sklearn.ensemble import RandomForestClassifier


# In[109]:


param_grid = {
    'max_depth': range(5,10,5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'n_estimators': [100,200,300], 
    'max_features': [10, 20]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, 
                           param_grid = param_grid, 
                           cv = 2,
                           n_jobs = -1,
                           verbose = 1, 
                           return_train_score=True)

# Fit the model
grid_search.fit(X_train, y_train)


# In[110]:


# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# In[95]:


# model with the best hyperparameters

rfc_imb_model = RandomForestClassifier(bootstrap=True,
                             max_depth=5,
                             min_samples_leaf=50, 
                             min_samples_split=50,
                             max_features=10,
                             n_estimators=100)


# In[96]:


# Fit the model
rfc_imb_model.fit(X_train, y_train)


# ##### Prediction on the train set

# In[97]:


# Predictions on the train set
y_train_pred = rfc_imb_model.predict(X_train)


# In[98]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train)
print(confusion)


# In[99]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[100]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train, y_train_pred))


# In[101]:


# classification_report
print(classification_report(y_train, y_train_pred))


# In[102]:


# Predicted probability
y_train_pred_proba = rfc_imb_model.predict_proba(X_train)[:,1]


# In[103]:


# roc_auc
auc = metrics.roc_auc_score(y_train, y_train_pred_proba)
auc


# In[104]:


# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba)


# ##### Prediction on the test set

# In[105]:


# Predictions on the test set
y_test_pred = rfc_imb_model.predict(X_test)


# In[106]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[107]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[108]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train, y_train_pred))


# In[109]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[110]:


# Predicted probability
y_test_pred_proba = rfc_imb_model.predict_proba(X_test)[:,1]


# In[111]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[112]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 1.0
#     - Specificity = 1.0
#     - F1-Score = 0.80
#     - ROC-AUC = 0.98
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.62
#     - Specificity = 0.99
#     - F-1 Score = 0.75
#     - ROC-AUC = 0.96

# ### Choosing best model on the imbalanced data
# 
# We can see that among all the models we tried (Logistic, XGBoost, Decision Tree, and Random Forest), almost all of them have performed well. More specifically Logistic regression and XGBoost performed best in terms of ROC-AUC score.
# 
# But as we have to choose one of them, we can go for the best as `XGBoost`, which gives us ROC score of 1.0 on the train data and 0.98 on the test data.
# 
# Keep in mind that XGBoost requires more resource utilization than Logistic model. Hence building XGBoost model is more costlier than the Logistic model. But XGBoost having ROC score 0.98, which is 0.01 more than the Logistic model. The 0.01 increase of score may convert into huge amount of saving for the bank.

# #### Print the important features of the best model to understand the dataset
# - This will not give much explanation on the already transformed dataset
# - But it will help us in understanding if the dataset is not PCA transformed

# In[57]:


# Features of XGBoost model

var_imp = []
for i in xgb_imb_model.feature_importances_:
    var_imp.append(i)
print('Top var =', var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-1])+1)
print('2nd Top var =', var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-2])+1)
print('3rd Top var =', var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-3])+1)
# Variable on Index-16 and Index-13 seems to be the top 2 variables
top_var_index = var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-1])
second_top_var_index = var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-2])

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]

np.random.shuffle(X_train_0)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [20, 20]

plt.scatter(X_train_1[:, top_var_index], X_train_1[:, second_top_var_index], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], top_var_index], X_train_0[:X_train_1.shape[0], second_top_var_index],
            label='Actual Class-0 Examples')
plt.legend()


# #### Print the FPR,TPR & select the best threshold from the roc curve for the best model

# In[66]:


print('Train auc =', metrics.roc_auc_score(y_train, y_train_pred_proba_imb_xgb))
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_pred_proba_imb_xgb)
threshold = thresholds[np.argmax(tpr-fpr)]
print("Threshold=",threshold)


# We can see that the threshold is 0.85, for which the TPR is the highest and FPR is the lowest and we got the best ROC score.

# # Handling data imbalance
# As we see that the data is heavily imbalanced, We will try several approaches for handling data imbalance.
# 
# - Undersampling :- Here for balancing the class distribution, the non-fraudulent transctions count will be reduced to 396 (similar count of fraudulent transctions)
# - Oversampling :- Here we will make the same count of non-fraudulent transctions as fraudulent transctions.
# - SMOTE :- Synthetic minority oversampling technique. It is another oversampling technique, which uses nearest neighbor algorithm to create synthetic data. 
# - Adasyn:- This is similar to SMOTE with minor changes that the new synthetic data is generated on the region of low density of imbalanced data points.

# ## Undersampling

# In[116]:


# Importing undersampler library
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# In[117]:


# instantiating the random undersampler 
rus = RandomUnderSampler()
# resampling X, y
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)


# In[118]:


# Befor sampling class distribution
print('Before sampling class distribution:-',Counter(y_train))
# new class distribution 
print('New class distribution:-',Counter(y_train_rus))


# ## Model building on balanced data with Undersampling

# ### Logistic Regression

# In[50]:


# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as roc-auc
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_rus, y_train_rus)


# In[51]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[52]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('roc_auc')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')


# In[53]:


# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test roc_auc is {0} at C = {1}".format(best_score, best_C))


# #### Logistic regression with optimal C

# In[119]:


# Instantiate the model with best C
logistic_bal_rus = LogisticRegression(C=0.1)


# In[120]:


# Fit the model on the train set
logistic_bal_rus_model = logistic_bal_rus.fit(X_train_rus, y_train_rus)


# ##### Prediction on the train set

# In[121]:


# Predictions on the train set
y_train_pred = logistic_bal_rus_model.predict(X_train_rus)


# In[122]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_rus, y_train_pred)
print(confusion)


# In[123]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[124]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_rus, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train_rus, y_train_pred))


# In[125]:


# classification_report
print(classification_report(y_train_rus, y_train_pred))


# In[126]:


# Predicted probability
y_train_pred_proba = logistic_bal_rus_model.predict_proba(X_train_rus)[:,1]


# In[127]:


# roc_auc
auc = metrics.roc_auc_score(y_train_rus, y_train_pred_proba)
auc


# In[128]:


# Plot the ROC curve
draw_roc(y_train_rus, y_train_pred_proba)


# #### Prediction on the test set

# In[129]:


# Prediction on the test set
y_test_pred = logistic_bal_rus_model.predict(X_test)


# In[130]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[131]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[132]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[133]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[134]:


# Predicted probability
y_test_pred_proba = logistic_bal_rus_model.predict_proba(X_test)[:,1]


# In[135]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[136]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.95
#     - Sensitivity = 0.92
#     - Specificity = 0.98
#     - ROC = 0.99
# - Test set
#     - Accuracy = 0.97
#     - Sensitivity = 0.86
#     - Specificity = 0.97
#     - ROC = 0.96

# ### XGBoost

# In[73]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train_rus, y_train_rus)       


# In[74]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[75]:


# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):


    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# ##### Model with optimal hyperparameters
# We see that the train score almost touches to 1. Among the hyperparameters, we can choose the best parameters as learning_rate : 0.2 and subsample: 0.3

# In[76]:


model_cv.best_params_


# In[137]:


# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for calculating auc
params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.6,
         'objective':'binary:logistic'}

# fit model on training data
xgb_bal_rus_model = XGBClassifier(params = params)
xgb_bal_rus_model.fit(X_train_rus, y_train_rus)


# ##### Prediction on the train set

# In[138]:


# Predictions on the train set
y_train_pred = xgb_bal_rus_model.predict(X_train_rus)


# In[139]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_rus, y_train_rus)
print(confusion)


# In[140]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[141]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_rus, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[142]:


# classification_report
print(classification_report(y_train_rus, y_train_pred))


# In[143]:


# Predicted probability
y_train_pred_proba = xgb_bal_rus_model.predict_proba(X_train_rus)[:,1]


# In[144]:


# roc_auc
auc = metrics.roc_auc_score(y_train_rus, y_train_pred_proba)
auc


# In[146]:


# Plot the ROC curve
draw_roc(y_train_rus, y_train_pred_proba)


# ##### Prediction on the test set

# In[147]:


# Predictions on the test set
y_test_pred = xgb_bal_rus_model.predict(X_test)


# In[148]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[149]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[150]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[151]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[152]:


# Predicted probability
y_test_pred_proba = xgb_bal_rus_model.predict_proba(X_test)[:,1]


# In[153]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[154]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 1.0
#     - Sensitivity = 1.0
#     - Specificity = 1.0
#     - ROC-AUC = 1.0
# - Test set
#     - Accuracy = 0.96
#     - Sensitivity = 0.92
#     - Specificity = 0.96
#     - ROC-AUC = 0.98

# ### Decision Tree

# In[105]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 3, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_rus,y_train_rus)


# In[106]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[107]:


# Printing the optimal sensitivity score and hyperparameters
print("Best roc_auc:-", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[155]:


# Model with optimal hyperparameters
dt_bal_rus_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)

dt_bal_rus_model.fit(X_train_rus, y_train_rus)


# ##### Prediction on the train set

# In[156]:


# Predictions on the train set
y_train_pred = dt_bal_rus_model.predict(X_train_rus)


# In[157]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_rus, y_train_pred)
print(confusion)


# In[158]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[159]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_rus, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[160]:


# classification_report
print(classification_report(y_train_rus, y_train_pred))


# In[161]:


# Predicted probability
y_train_pred_proba = dt_bal_rus_model.predict_proba(X_train_rus)[:,1]


# In[162]:


# roc_auc
auc = metrics.roc_auc_score(y_train_rus, y_train_pred_proba)
auc


# In[163]:


# Plot the ROC curve
draw_roc(y_train_rus, y_train_pred_proba)


# ##### Prediction on the test set

# In[164]:


# Predictions on the test set
y_test_pred = dt_bal_rus_model.predict(X_test)


# In[165]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[166]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[167]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[168]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[169]:


# Predicted probability
y_test_pred_proba = dt_bal_rus_model.predict_proba(X_test)[:,1]


# In[170]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[171]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.93
#     - Sensitivity = 0.88
#     - Specificity = 0.97
#     - ROC-AUC = 0.98
# - Test set
#     - Accuracy = 0.96
#     - Sensitivity = 0.85
#     - Specificity = 0.96
#     - ROC-AUC = 0.96

# ### Random forest

# In[123]:


param_grid = {
    'max_depth': range(5,10,5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'n_estimators': [100,200,300], 
    'max_features': [10, 20]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 2,
                           n_jobs = -1,
                           verbose = 1, 
                           return_train_score=True)

# Fit the model
grid_search.fit(X_train_rus, y_train_rus)


# In[124]:


# printing the optimal accuracy score and hyperparameters
print('We can get roc-auc of',grid_search.best_score_,'using',grid_search.best_params_)


# In[172]:


# model with the best hyperparameters

rfc_bal_rus_model = RandomForestClassifier(bootstrap=True,
                             max_depth=5,
                             min_samples_leaf=50, 
                             min_samples_split=50,
                             max_features=10,
                             n_estimators=200)


# In[173]:


# Fit the model
rfc_bal_rus_model.fit(X_train_rus, y_train_rus)


# ##### Prediction on the train set

# In[174]:


# Predictions on the train set
y_train_pred = rfc_bal_rus_model.predict(X_train_rus)


# In[175]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_rus, y_train_pred)
print(confusion)


# In[176]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[177]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_rus, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train_rus, y_train_pred))


# In[178]:


# classification_report
print(classification_report(y_train_rus, y_train_pred))


# In[179]:


# Predicted probability
y_train_pred_proba = rfc_bal_rus_model.predict_proba(X_train_rus)[:,1]


# In[180]:


# roc_auc
auc = metrics.roc_auc_score(y_train_rus, y_train_pred_proba)
auc


# In[181]:


# Plot the ROC curve
draw_roc(y_train_rus, y_train_pred_proba)


# ##### Prediction on the test set

# In[182]:


# Predictions on the test set
y_test_pred = rfc_bal_rus_model.predict(X_test)


# In[183]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[184]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[185]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[186]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[187]:


# Predicted probability
y_test_pred_proba = rfc_bal_rus_model.predict_proba(X_test)[:,1]


# In[188]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[189]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.94
#     - Sensitivity = 0.89
#     - Specificity = 0.98
#     - ROC-AUC = 0.98
# - Test set
#     - Accuracy = 0.98
#     - Sensitivity = 0.83
#     - Specificity = 0.98
#     - ROC-AUC = 0.97

# # Oversampling

# In[190]:


# Importing oversampler library
from imblearn.over_sampling import RandomOverSampler


# In[191]:


# instantiating the random oversampler 
ros = RandomOverSampler()
# resampling X, y
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)


# In[192]:


# Befor sampling class distribution
print('Before sampling class distribution:-',Counter(y_train))
# new class distribution 
print('New class distribution:-',Counter(y_train_ros))


# ### Logistic Regression

# In[145]:


# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as roc-auc
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_ros, y_train_ros)


# In[146]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[147]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('roc_auc')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')


# In[148]:


# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test roc_auc is {0} at C = {1}".format(best_score, best_C))


# #### Logistic regression with optimal C

# In[193]:


# Instantiate the model with best C
logistic_bal_ros = LogisticRegression(C=0.1)


# In[194]:


# Fit the model on the train set
logistic_bal_ros_model = logistic_bal_ros.fit(X_train_ros, y_train_ros)


# ##### Prediction on the train set

# In[195]:


# Predictions on the train set
y_train_pred = logistic_bal_ros_model.predict(X_train_ros)


# In[196]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_ros, y_train_pred)
print(confusion)


# In[197]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[198]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_ros, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train_ros, y_train_pred))


# In[199]:


# classification_report
print(classification_report(y_train_ros, y_train_pred))


# In[200]:


# Predicted probability
y_train_pred_proba = logistic_bal_ros_model.predict_proba(X_train_ros)[:,1]


# In[201]:


# roc_auc
auc = metrics.roc_auc_score(y_train_ros, y_train_pred_proba)
auc


# In[202]:


# Plot the ROC curve
draw_roc(y_train_ros, y_train_pred_proba)


# #### Prediction on the test set

# In[203]:


# Prediction on the test set
y_test_pred = logistic_bal_ros_model.predict(X_test)


# In[204]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[205]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[206]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[207]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[208]:


# Predicted probability
y_test_pred_proba = logistic_bal_ros_model.predict_proba(X_test)[:,1]


# In[209]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[210]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.95
#     - Sensitivity = 0.92
#     - Specificity = 0.97
#     - ROC = 0.98
# - Test set
#     - Accuracy = 0.97
#     - Sensitivity = 0.89
#     - Specificity = 0.97
#     - ROC = 0.97

# ### XGBoost

# In[222]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train_ros, y_train_ros)       


# In[164]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[165]:


# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):


    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# ##### Model with optimal hyperparameters
# We see that the train score almost touches to 1. Among the hyperparameters, we can choose the best parameters as learning_rate : 0.2 and subsample: 0.3

# In[166]:


model_cv.best_params_


# In[211]:


# chosen hyperparameters
params = {'learning_rate': 0.6,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
xgb_bal_ros_model = XGBClassifier(params = params)
xgb_bal_ros_model.fit(X_train_ros, y_train_ros)


# ##### Prediction on the train set

# In[212]:


# Predictions on the train set
y_train_pred = xgb_bal_ros_model.predict(X_train_ros)


# In[213]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_ros, y_train_ros)
print(confusion)


# In[214]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[215]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_ros, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[216]:


# classification_report
print(classification_report(y_train_ros, y_train_pred))


# In[217]:


# Predicted probability
y_train_pred_proba = xgb_bal_ros_model.predict_proba(X_train_ros)[:,1]


# In[218]:


# roc_auc
auc = metrics.roc_auc_score(y_train_ros, y_train_pred_proba)
auc


# In[219]:


# Plot the ROC curve
draw_roc(y_train_ros, y_train_pred_proba)


# ##### Prediction on the test set

# In[223]:


# Predictions on the test set
y_test_pred = xgb_bal_ros_model.predict(X_test)


# In[224]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[225]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[226]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[227]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[228]:


# Predicted probability
y_test_pred_proba = xgb_bal_ros_model.predict_proba(X_test)[:,1]


# In[229]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[230]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 1.0
#     - Sensitivity = 1.0
#     - Specificity = 1.0
#     - ROC-AUC = 1.0
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.80
#     - Specificity = 0.99
#     - ROC-AUC = 0.97

# ### Decision Tree

# In[180]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 3, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_ros,y_train_ros)


# In[181]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[182]:


# Printing the optimal sensitivity score and hyperparameters
print("Best roc_auc:-", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[231]:


# Model with optimal hyperparameters
dt_bal_ros_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=100,
                                  min_samples_split=50)

dt_bal_ros_model.fit(X_train_ros, y_train_ros)


# ##### Prediction on the train set

# In[232]:


# Predictions on the train set
y_train_pred = dt_bal_ros_model.predict(X_train_ros)


# In[233]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_ros, y_train_pred)
print(confusion)


# In[234]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[235]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_ros, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[236]:


# classification_report
print(classification_report(y_train_ros, y_train_pred))


# In[237]:


# Predicted probability
y_train_pred_proba = dt_bal_ros_model.predict_proba(X_train_ros)[:,1]


# In[238]:


# roc_auc
auc = metrics.roc_auc_score(y_train_ros, y_train_pred_proba)
auc


# In[239]:


# Plot the ROC curve
draw_roc(y_train_ros, y_train_pred_proba)


# ##### Prediction on the test set

# In[240]:


# Predictions on the test set
y_test_pred = dt_bal_ros_model.predict(X_test)


# In[241]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[242]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[243]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[244]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[245]:


# Predicted probability
y_test_pred_proba = dt_bal_ros_model.predict_proba(X_test)[:,1]


# In[246]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[247]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 1.0
#     - Specificity = 0.99
#     - ROC-AUC = 0.99
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.79
#     - Specificity = 0.99
#     - ROC-AUC = 0.90

# ## SMOTE (Synthetic Minority Oversampling Technique)

# We are creating synthetic samples by doing upsampling using SMOTE(Synthetic Minority Oversampling Technique).

# In[68]:


# Importing SMOTE
from imblearn.over_sampling import SMOTE


# In[69]:


# Instantiate SMOTE
sm = SMOTE(random_state=27)
# Fitting SMOTE to the train set
X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)


# In[70]:


print('Before SMOTE oversampling X_train shape=',X_train.shape)
print('After SMOTE oversampling X_train shape=',X_train_smote.shape)


# ### Logistic Regression

# In[ ]:


# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as roc-auc
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_smote, y_train_smote)


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('roc_auc')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')


# In[ ]:


# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test roc_auc is {0} at C = {1}".format(best_score, best_C))


# #### Logistic regression with optimal C

# In[71]:


# Instantiate the model with best C
logistic_bal_smote = LogisticRegression(C=0.1)


# In[72]:


# Fit the model on the train set
logistic_bal_smote_model = logistic_bal_smote.fit(X_train_smote, y_train_smote)


# ##### Prediction on the train set

# In[73]:


# Predictions on the train set
y_train_pred = logistic_bal_smote_model.predict(X_train_smote)


# In[74]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_smote, y_train_pred)
print(confusion)


# In[75]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[76]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_smote, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[77]:


# classification_report
print(classification_report(y_train_smote, y_train_pred))


# In[89]:


# Predicted probability
y_train_pred_proba_log_bal_smote = logistic_bal_smote_model.predict_proba(X_train_smote)[:,1]


# In[90]:


# Plot the ROC curve
draw_roc(y_train_smote, y_train_pred_proba_log_bal_smote)


# #### Prediction on the test set

# In[80]:


# Prediction on the test set
y_test_pred = logistic_bal_smote_model.predict(X_test)


# In[81]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[82]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[83]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[84]:


# classification_report
print(classification_report(y_test, y_test_pred))


# ##### ROC on the test set

# In[85]:


# Predicted probability
y_test_pred_proba = logistic_bal_smote_model.predict_proba(X_test)[:,1]


# In[86]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.95
#     - Sensitivity = 0.92
#     - Specificity = 0.98
#     - ROC = 0.99
# - Test set
#     - Accuracy = 0.97
#     - Sensitivity = 0.90
#     - Specificity = 0.99
#     - ROC = 0.97

# ### XGBoost

# In[ ]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train_smote, y_train_smote)       


# In[ ]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):


    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# ##### Model with optimal hyperparameters
# We see that the train score almost touches to 1. Among the hyperparameters, we can choose the best parameters as learning_rate : 0.2 and subsample: 0.3

# In[ ]:


model_cv.best_params_


# In[267]:


# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for calculating auc
params = {'learning_rate': 0.6,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
xgb_bal_smote_model = XGBClassifier(params = params)
xgb_bal_smote_model.fit(X_train_smote, y_train_smote)


# ##### Prediction on the train set

# In[268]:


# Predictions on the train set
y_train_pred = xgb_bal_smote_model.predict(X_train_smote)


# In[269]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_smote, y_train_pred)
print(confusion)


# In[270]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[271]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_smote, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[272]:


# classification_report
print(classification_report(y_train_smote, y_train_pred))


# In[273]:


# Predicted probability
y_train_pred_proba = xgb_bal_smote_model.predict_proba(X_train_smote)[:,1]


# In[274]:


# roc_auc
auc = metrics.roc_auc_score(y_train_smote, y_train_pred_proba)
auc


# In[275]:


# Plot the ROC curve
draw_roc(y_train_smote, y_train_pred_proba)


# ##### Prediction on the test set

# In[276]:


# Predictions on the test set
y_test_pred = xgb_bal_smote_model.predict(X_test)


# In[277]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[278]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[279]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[280]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[281]:


# Predicted probability
y_test_pred_proba = xgb_bal_smote_model.predict_proba(X_test)[:,1]


# In[282]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[283]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 1.0
#     - Specificity = 0.99
#     - ROC-AUC = 1.0
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.79
#     - Specificity = 0.99
#     - ROC-AUC = 0.96
# 
# Overall, the model is performing well in the test set, what it had learnt from the train set.

# ### Decision Tree

# In[ ]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 3, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_smote,y_train_smote)


# In[ ]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[ ]:


# Printing the optimal sensitivity score and hyperparameters
print("Best roc_auc:-", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[284]:


# Model with optimal hyperparameters
dt_bal_smote_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=100)

dt_bal_smote_model.fit(X_train_smote, y_train_smote)


# ##### Prediction on the train set

# In[285]:


# Predictions on the train set
y_train_pred = dt_bal_smote_model.predict(X_train_smote)


# In[286]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_smote, y_train_pred)
print(confusion)


# In[287]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[288]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_smote, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[290]:


# classification_report
print(classification_report(y_train_smote, y_train_pred))


# In[291]:


# Predicted probability
y_train_pred_proba = dt_bal_smote_model.predict_proba(X_train_smote)[:,1]


# In[292]:


# roc_auc
auc = metrics.roc_auc_score(y_train_smote, y_train_pred_proba)
auc


# In[294]:


# Plot the ROC curve
draw_roc(y_train_smote, y_train_pred_proba)


# ##### Prediction on the test set

# In[295]:


# Predictions on the test set
y_test_pred = dt_bal_smote_model.predict(X_test)


# In[296]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[297]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[298]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[299]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[300]:


# Predicted probability
y_test_pred_proba = dt_bal_smote_model.predict_proba(X_test)[:,1]


# In[301]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[302]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 0.99
#     - Specificity = 0.98
#     - ROC-AUC = 0.99
# - Test set
#     - Accuracy = 0.98
#     - Sensitivity = 0.80
#     - Specificity = 0.98
#     - ROC-AUC = 0.86
# 

# ## AdaSyn (Adaptive Synthetic Sampling)

# In[303]:


# Importing adasyn
from imblearn.over_sampling import ADASYN


# In[304]:


# Instantiate adasyn
ada = ADASYN(random_state=0)
X_train_adasyn, y_train_adasyn = ada.fit_resample(X_train, y_train)


# In[305]:


# Befor sampling class distribution
print('Before sampling class distribution:-',Counter(y_train))
# new class distribution 
print('New class distribution:-',Counter(y_train_adasyn))


# ### Logistic Regression

# In[238]:


# Creating KFold object with 3 splits
folds = KFold(n_splits=3, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as roc-auc
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_adasyn, y_train_adasyn)


# In[239]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[240]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('roc_auc')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')


# In[241]:


# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test roc_auc is {0} at C = {1}".format(best_score, best_C))


# #### Logistic regression with optimal C

# In[306]:


# Instantiate the model with best C
logistic_bal_adasyn = LogisticRegression(C=1000)


# In[307]:


# Fit the model on the train set
logistic_bal_adasyn_model = logistic_bal_adasyn.fit(X_train_adasyn, y_train_adasyn)


# ##### Prediction on the train set

# In[308]:


# Predictions on the train set
y_train_pred = logistic_bal_adasyn_model.predict(X_train_adasyn)


# In[309]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_adasyn, y_train_pred)
print(confusion)


# In[310]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[311]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_adasyn, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

# F1 score
print("F1-Score:-", f1_score(y_train_adasyn, y_train_pred))


# In[312]:


# classification_report
print(classification_report(y_train_adasyn, y_train_pred))


# In[313]:


# Predicted probability
y_train_pred_proba = logistic_bal_adasyn_model.predict_proba(X_train_adasyn)[:,1]


# In[314]:


# roc_auc
auc = metrics.roc_auc_score(y_train_adasyn, y_train_pred_proba)
auc


# In[315]:


# Plot the ROC curve
draw_roc(y_train_adasyn, y_train_pred_proba)


# #### Prediction on the test set

# In[316]:


# Prediction on the test set
y_test_pred = logistic_bal_adasyn_model.predict(X_test)


# In[317]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[318]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[319]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[320]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[321]:


# Predicted probability
y_test_pred_proba = logistic_bal_adasyn_model.predict_proba(X_test)[:,1]


# In[322]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[323]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.88
#     - Sensitivity = 0.86
#     - Specificity = 0.91
#     - ROC = 0.96
# - Test set
#     - Accuracy = 0.90
#     - Sensitivity = 0.95
#     - Specificity = 0.90
#     - ROC = 0.97

# ### Decision Tree

# In[205]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 3, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_adasyn,y_train_adasyn)


# In[206]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[207]:


# Printing the optimal sensitivity score and hyperparameters
print("Best roc_auc:-", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[324]:


# Model with optimal hyperparameters
dt_bal_adasyn_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=100,
                                  min_samples_split=50)

dt_bal_adasyn_model.fit(X_train_adasyn, y_train_adasyn)


# ##### Prediction on the train set

# In[325]:


# Predictions on the train set
y_train_pred = dt_bal_adasyn_model.predict(X_train_adasyn)


# In[326]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_adasyn, y_train_pred)
print(confusion)


# In[327]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[328]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_adasyn, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[329]:


# classification_report
print(classification_report(y_train_adasyn, y_train_pred))


# In[330]:


# Predicted probability
y_train_pred_proba = dt_bal_adasyn_model.predict_proba(X_train_adasyn)[:,1]


# In[331]:


# roc_auc
auc = metrics.roc_auc_score(y_train_adasyn, y_train_pred_proba)
auc


# In[332]:


# Plot the ROC curve
draw_roc(y_train_adasyn, y_train_pred_proba)


# ##### Prediction on the test set

# In[333]:


# Predictions on the test set
y_test_pred = dt_bal_adasyn_model.predict(X_test)


# In[334]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[335]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[336]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[337]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[338]:


# Predicted probability
y_test_pred_proba = dt_bal_adasyn_model.predict_proba(X_test)[:,1]


# In[339]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[340]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.97
#     - Sensitivity = 0.99
#     - Specificity = 0.95
#     - ROC-AUC = 0.99
# - Test set
#     - Accuracy = 0.95
#     - Sensitivity = 0.84
#     - Specificity = 0.95
#     - ROC-AUC = 0.91

# ### XGBoost

# In[221]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train_adasyn, y_train_adasyn)       


# In[222]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[223]:


# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):


    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[224]:


model_cv.best_params_


# In[341]:


# chosen hyperparameters

params = {'learning_rate': 0.6,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.3,
         'objective':'binary:logistic'}

# fit model on training data
xgb_bal_adasyn_model = XGBClassifier(params = params)
xgb_bal_adasyn_model.fit(X_train_adasyn, y_train_adasyn)


# ##### Prediction on the train set

# In[342]:


# Predictions on the train set
y_train_pred = xgb_bal_adasyn_model.predict(X_train_adasyn)


# In[343]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train_adasyn, y_train_adasyn)
print(confusion)


# In[344]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[345]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_adasyn, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[346]:


# classification_report
print(classification_report(y_train_adasyn, y_train_pred))


# In[347]:


# Predicted probability
y_train_pred_proba = xgb_bal_adasyn_model.predict_proba(X_train_adasyn)[:,1]


# In[348]:


# roc_auc
auc = metrics.roc_auc_score(y_train_adasyn, y_train_pred_proba)
auc


# In[349]:


# Plot the ROC curve
draw_roc(y_train_adasyn, y_train_pred_proba)


# ##### Prediction on the test set

# In[350]:


# Predictions on the test set
y_test_pred = xgb_bal_adasyn_model.predict(X_test)


# In[351]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[352]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[353]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[354]:


# classification_report
print(classification_report(y_test, y_test_pred))


# In[355]:


# Predicted probability
y_test_pred_proba = xgb_bal_adasyn_model.predict_proba(X_test)[:,1]


# In[356]:


# roc_auc
auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
auc


# In[357]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.99
#     - Sensitivity = 1.0
#     - Specificity = 1.0
#     - ROC-AUC = 1.0
# - Test set
#     - Accuracy = 0.99
#     - Sensitivity = 0.78
#     - Specificity = 0.99
#     - ROC-AUC = 0.96

# ### Choosing best model on the balanced data
# 
# He we balanced the data with various approach such as Undersampling, Oversampling, SMOTE and Adasy. With every data balancing thechnique we built several models such as Logistic, XGBoost, Decision Tree, and Random Forest.
# 
# We can see that almost all the models performed more or less good. But we should be interested in the best model. 
# 
# Though the Undersampling technique models performed well, we should keep mind that by doing the undersampling some imformation were lost. Hence, it is better not to consider the undersampling models.
# 
# Whereas the SMOTE and Adasyn models performed well. Among those models the simplist model Logistic regression has ROC score 0.99 in the train set and 0.97 on the test set. We can consider the Logistic model as the best model to choose because of the easy interpretation of the models and also the resourse requirements to build the mdoel is lesser than the other heavy models such as Random forest or XGBoost.
# 
# Hence, we can conclude that the `Logistic regression model with SMOTE` is the best model for its simlicity and less resource requirement. 

# #### Print the FPR,TPR & select the best threshold from the roc curve for the best model

# In[92]:


print('Train auc =', metrics.roc_auc_score(y_train_smote, y_train_pred_proba_log_bal_smote))
fpr, tpr, thresholds = metrics.roc_curve(y_train_smote, y_train_pred_proba_log_bal_smote)
threshold = thresholds[np.argmax(tpr-fpr)]
print("Threshold=",threshold)


# We can see that the threshold is 0.53, for which the TPR is the highest and FPR is the lowest and we got the best ROC score.

# ## Cost benefit analysis
# We have tried several models till now with both balanced and imbalanced data. We have noticed most of the models have performed more or less well in terms of ROC score, Precision and Recall.
# 
# But while picking the best model we should consider few things such as whether we have required infrastructure, resources or computational power to run the model or not. For the models such as Random forest, SVM, XGBoost we require heavy computational resources and eventually to build that infrastructure the cost of deploying the model increases. On the other hand the simpler model such as Logistic regression requires less computational resources, so the cost of building the model is less.
# 
# We also have to consider that for little change of the ROC score how much monetary loss of gain the bank incur. If the amount if huge then we have to consider building the complex model even though the cost of building the model is high. 

# ## Summary to the business
# For banks with smaller average transaction value, we would want high precision because we only want to label relevant transactions as fraudulent. For every transaction that is flagged as fraudulent, we can add the human element to verify whether the transaction was done by calling the customer. However, when precision is low, such tasks are a burden because the human element has to be increased.
# 
# For banks having a larger transaction value, if the recall is low, i.e., it is unable to detect transactions that are labelled as non-fraudulent. So we have to consider the losses if the missed transaction was a high-value fraudulent one.
# 
# So here, to save the banks from high-value fraudulent transactions, we have to focus on a high recall in order to detect actual fraudulent transactions.
# 
# After performing several models, we have seen that in the balanced dataset with SMOTE technique the simplest Logistic regression model has good ROC score and also high Recall. Hence, we can go with the logistic model here. It is also easier to interpret and explain to the business.
