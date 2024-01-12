# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python
# coding: utf-8
#ML challenge groups 40

# Import libraries

# In[51]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


import warnings

warnings.filterwarnings('ignore')


# Import dataset

# In[53]:


data = 'Newdata-2.csv'

df = pd.read_csv(data)


# Exploratory data analysis

# In[54]:


# view dimensions of dataset

df.shape


# In[55]:


# preview the dataset

df.head()


# View summary of dataset

# In[56]:


df.info()


# Frequency distribution of values in variables

# In[57]:


for col in df[0:]:
    
    print(df[col].value_counts())


# Explore class variable

# In[58]:


df['Transaction'].value_counts()


# Missing values in variables

# In[59]:


df.isnull().sum()


# Declare feature vector and target variable

# In[60]:


X = df.drop(['Transaction'], axis=1)

y = df['Transaction']


import matplotlib.pyplot as plt

# Calculate the sizes of each target value
sizes = [len(y[y == 0]), len(y[y == 1])]

# Data for the pie plot
labels = ['label 0', 'label 1']
colors = ['blue', 'orange']

# Create the pie plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

# Add a title
plt.title('Pie Plot of the Frequency of Two Target Values')

# Display the plot
plt.show()





# Split data into separate training and test set

# In[61]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[62]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[63]:


import pandas as pd
print("targets in train set")
counts_y_train = pd.Series(y_train).value_counts()
print(counts_y_train)
print()
print("targets in test set")
counts_y_test = pd.Series(y_test).value_counts()
print(counts_y_test)


# Feature Engineering 

# In[64]:


# check data types in X_train

X_train.dtypes


# Encode categorical variables

# In[65]:


X_train.head()




# In[67]:


import category_encoders as ce


# Display categorical variables in training set

# In[68]:


categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# Display numerical variables in training set

# In[69]:


numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[70]:


# encode categorical variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['Customer_Type'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[71]:


X_train.head()


# In[72]:


X_test.head()


#  Feature Scaling

# In[73]:


cols = X_train.columns


# In[74]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[75]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[76]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[77]:


# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier



# instantiate the classifier 

rfc = RandomForestClassifier(random_state=0)



# fit the model

rfc.fit(X_train, y_train)



# Check accuracy score 

from sklearn.metrics import accuracy_score, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV

#hyperparameters
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score)
}

best_rf = GridSearchCV(rfc, {'n_estimators' : [100,200,300]}, cv=5, scoring=scoring, refit='accuracy')
best_rf.fit(X_train, y_train)

best_model_rf = best_rf.best_estimator_



# fit the model to the training set
best_model_rf.fit(X_train, y_train)




# Predict on the test set results

y_pred_rf = best_model_rf.predict(X_test)



# Check accuracy score 

print('Model accuracy score of Random Forest : {0:0.4f}'. format(accuracy_score(y_test, y_pred_rf)))

print('Best Parameters:', best_rf.best_params_)
print('Accuracy Score:', best_rf.best_score_)
print('F1 Score:', f1_score(y_test, y_pred_rf))




# Confusion matrix

# In[79]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)

print('Confusion matrix\n\n', cm)


# In[80]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rf))


# In[82]:


counts_y_perd_rf = pd.Series(y_pred_rf).value_counts()
print(counts_y_perd_rf)


# using XGBooster Classifier

# In[83]:


get_ipython().run_line_magic('pip', 'install xgboost')
from xgboost import XGBClassifier
import xgboost as xgb


# In[84]:


xgbootmodel = XGBClassifier(learning_rate=0.1,n_estimators=300,booster="gbtree",reg_lambda=0.5,reg_alpha=0.5)

#hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'reg_lambda': [0.1, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0]
}
scoring = ['accuracy', 'f1']

best_xgb = GridSearchCV(xgbootmodel, param_grid, cv=5, scoring=scoring, refit='accuracy')
best_xgb.fit(X_train, y_train)

best_model_xgb = best_xgb.best_estimator_



# fit the model to the training set
best_model_xgb.fit(X_train, y_train)


#  Find important features with Random Forest model 
# 
#  Visualize feature scores of the features
import matplotlib.pyplot as plt

# Get the best estimator from GridSearchCV
best_estimator = best_xgb.best_estimator_

# Get feature importances
feature_importances = pd.Series(best_estimator.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances of XGB model')
plt.show()

# In[85]:


y_pred_xgb = best_model_xgb.predict(X_test)

print('Best Parameters:', best_xgb.best_params_)
print('Accuracy Score:', best_xgb.best_score_)
print('F1 Score:', f1_score(y_test, y_pred_xgb))



# save the predication


df_prediction = pd.DataFrame(y_pred_xgb, columns=['predication'])

shape1 = df_prediction.shape

print(shape1)

# Save the DataFrame as a CSV file
df_prediction.to_csv('predicted_data.csv', index=False)


# In[86]:


cmxgb = confusion_matrix(y_test, y_pred_xgb)

print('Confusion matrix\n\n', cmxgb)


# In[87]:


print(classification_report(y_test, y_pred_xgb))


# Logistic Regression with Sklearn

# Scaling the Features

# In[88]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()


# In[89]:


scaler.fit(X_train)


# In[90]:


scaled_X_train= scaler.transform(X_train)
scaled_X_test= scaler.transform(X_test)


# Train the Model

# In[91]:


from sklearn.linear_model import LogisticRegression
log_model= LogisticRegression()
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

best_log = GridSearchCV(log_model, param_grid, cv=5, scoring='accuracy', refit=True)
best_log.fit(scaled_X_train, y_train)


# In[92]:


best_log.best_estimator_.coef_

best_model_log = best_log.best_estimator_


# Predicting Test Data

# In[93]:


y_pred_log= best_model_log.predict(scaled_X_test)

print('Best Parameters:', best_log.best_params_)
print('Accuracy Score:', best_log.best_score_)
print('F1 Score:', f1_score(y_test, y_pred_log))


# Evaluating the Model

# In[94]:


confusion_matrix(y_test, y_pred_log)


# In[95]:


print(classification_report(y_test, y_pred_log))


# classification with neural networks

# In[96]:


get_ipython().system('pip install tensorflow')
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy  as np
from tensorflow import keras
import matplotlib.pyplot as plt


# In[97]:


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train)
X_test_nn = scaler.transform(X_test)
y_train_nn = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
y_test_nn = scaler.transform(np.array(y_test).reshape(-1, 1))


# In[98]:


from tensorflow.keras.regularizers import l2


# In[99]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', kernel_regularizer=l2(0.01), input_dim = X_train_nn.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', kernel_regularizer=l2(0.01)))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid' , kernel_regularizer=l2(0.01)))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train_nn, y = y_train_nn, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# In[100]:



classifier.fit(X_train, y_train)


# In[101]:


y_pred_nn = classifier.predict(X_test)


# In[102]:


accuracy_score(y_test, y_pred_nn)


# In[103]:


confusion_matrix(y_test, y_pred_nn)


# In[104]:


print(classification_report(y_test, y_pred_nn))


# Support Vector Machines

# In[105]:


from sklearn.svm import SVC



svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=0)

svm_model.fit(X_train, y_train)

scoring = ['accuracy', 'f1']

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto']
}


best_svm = GridSearchCV(svm_model, param_grid, cv=5, scoring=scoring, refit='accuracy')
best_svm.fit(X_train, y_train)

best_model_svm = best_svm.best_estimator_



# fit the model to the training set
best_model_svm.fit(X_train, y_train)



y_pred_svm = best_model_svm.predict(X_test)




# In[106]:


confusion_matrix(y_test, y_pred_svm)


# In[107]:


print(classification_report(y_test, y_pred_svm))


# Storing machine learning algorithms (MLA) in a variable

# In[112]:


# Application of all Machine Learning methods
models = []
models.append(('LOG', best_model_log))
models.append(('NN', classifier))
models.append(('RF', best_model_rf))
models.append(('XGB', best_model_xgb))
models.append(('SVM', best_model_svm))


# In[113]:


from sklearn import model_selection
from sklearn.metrics import accuracy_score


# In[114]:



seed = 42


# In[115]:


# evaluate each model in turn
from sklearn.model_selection import cross_validate

scoring = ['f1', 'accuracy']
for score in scoring:
    results = []
    names = []
    for name, model in models:
        cv_results = cross_validate(model, X_train, y_train, cv=10, scoring=score)
        results.append(cv_results['test_score'])
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results['test_score'].mean(), cv_results['test_score'].std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Comparison between different MLAs - ' + score + ' score')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()




