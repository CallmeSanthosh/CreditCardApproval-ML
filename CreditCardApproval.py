#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all the dependent python libraries for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, balanced_accuracy_score, plot_precision_recall_curve, precision_recall_curve, average_precision_score


# In[2]:


# Load the data using panda's read_csv method and verify whether the data has been loaded properly or not using sample method
# set header to None. This will prevent first row acting as a column name.
credit_data = pd.read_csv('crx.csv', header=None)
credit_data.sample(5)


# In[3]:


# Set the column names.
credit_data.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
credit_data.sample(5)


# In[4]:


# Info method is used to inspect the data types and missing values. With the result below, we found there is no null value present in the dataset.
# However, the datatype does not match with the requirement.
credit_data.info()


# In[5]:


# Cross verify null values presence in the dataset. There is no null value present in the dataset.
print(credit_data.isnull().values.sum())


# In[6]:


# On Manual inspection, '?' is the only invalid value in the whole dataset.
# So, Instead of removing the rows that have '?s', Replace '?' with NaN value.
credit_data = credit_data.replace('?', np.nan)

# Inspect the changes by applying boolean mask technique
credit_data[~credit_data['A1'].notna()]


# In[7]:


credit_data.info()


# In[8]:


# Drop all the nan values from the dataset
credit_data.dropna(inplace=True)
credit_data.info()


# In[9]:


# Set appropriate dtype for numerical columns 
credit_data["A2"] = pd.to_numeric(credit_data["A2"], downcast='float')
credit_data["A14"] = pd.to_numeric(credit_data["A14"], downcast='integer')


# In[10]:


# Visualize target data balance
sb.countplot(x = 'A16', data = credit_data, palette = 'Set3')


# In[11]:


# Store object dtype column names other than target variable 'A16'. This will be useful in HOT Encoder section
credit_data_catagorial = credit_data.select_dtypes(include=[object]).columns[0:-1]
credit_data_catagorial


# In[12]:


# Using label encoder to convert non numerical data into numeric types
LE = LabelEncoder()
for col in credit_data:
    if credit_data[col].dtypes=='object':
        credit_data[col]=LE.fit_transform(credit_data[col])


# In[13]:


# HOT ENCODER converts the each catagorial data column to multiple binary data dummy columns. This helps to weigh the values properly.
credit_encoded = pd.get_dummies(credit_data, columns=credit_data_catagorial)
credit_encoded.sample(5)


# In[14]:


# Segregate features and labels into separate variables
X, y = credit_encoded.iloc[:,credit_encoded.columns != 'A16'] , credit_encoded['A16']

# Split into train and test sets into 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, stratify=y)


# In[15]:


# Scaling X_train and X_test using MinMaxScalar to fit the data in logistic regression algorithm.
scaler = MinMaxScaler(feature_range=(0, 1))
scaledX_train = scaler.fit_transform(X_train)
scaledX_test = scaler.transform(X_test)


# In[16]:


# Apply logistic regression algorithm
logreg = LogisticRegression(class_weight=None, penalty='none')
logreg.fit(scaledX_train, y_train)

y_pred = logreg.predict(scaledX_test)
print('Sample Output of Prediction is:', y_pred[0:4]) 

Decision_Function = logreg.decision_function(scaledX_test) 
print('Sample Output of Decision Function is:', Decision_Function[0:4])

pred_pro = logreg.predict_proba(scaledX_test)
print('Sample Output of Predict Proba Function is:', pred_pro[0:4,:])


# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(scaledX_test, y_test))

# Get the balance accuracy score of logreg model and print it
bal_accuracy = balanced_accuracy_score(y_test, y_pred)
print('balance accuracy score: {:.4f}'.format(bal_accuracy))

# Print the confusion matrix of the logreg model
conf_mat = confusion_matrix(y_pred, y_test)
mat_names = ['True Neg','False Pos','False Neg','True Pos']
mat_counts = ['{0:0.0f}'.format(value) for value in
                conf_mat.flatten()]
mat_percentages = ['{0:.2%}'.format(value) for value in
                     conf_mat.flatten()/np.sum(conf_mat)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(mat_names,mat_counts,mat_percentages)]
labels = np.asarray(labels).reshape(2,2)
sb.heatmap(conf_mat, annot=labels, fmt='', cmap='Blues')

# Print classification report
target_names = ['Credit not approved', 'Credit Approved']
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=target_names))


# In[17]:


# Apply logistic regression algorithm with balanced weight
logreg_bal = LogisticRegression(class_weight='balanced', penalty='none')
logreg_bal.fit(scaledX_train, y_train)

y_pred_bal = logreg_bal.predict(scaledX_test)
print('Sample Output of Prediction is:', y_pred_bal[0:4]) 

Decision_Function_bal = logreg_bal.decision_function(scaledX_test) 
print('Sample Output of Balanced weight Decision Function is:', Decision_Function_bal[0:4])

pred_pro_bal = logreg_bal.predict_proba(scaledX_test)
print('Sample Output of Balanced weightPredict Proba Function is:', pred_pro_bal[0:4,:])

bal_accuracy_bal = balanced_accuracy_score(y_test, y_pred_bal)
print('balance accuracy score: {:.4f}'.format(bal_accuracy_bal))

# Get the accuracy score of logreg model with balanced weight and print it
print("Accuracy of logistic regression classifier Balanced weight: ", logreg_bal.score(scaledX_test, y_test))

# Print the confusion matrix of the logreg model with balanced weight
confusion_matrix(y_pred_bal, y_test)

conf_mat_bal = confusion_matrix(y_pred_bal, y_test)

mat_names_bal = ['True Neg','False Pos','False Neg','True Pos']
mat_counts_bal = ['{0:0.0f}'.format(value) for value in
                conf_mat_bal.flatten()]
mat_percentages_bal = ['{0:.2%}'.format(value) for value in
                     conf_mat_bal.flatten()/np.sum(conf_mat_bal)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(mat_names_bal,mat_counts_bal,mat_percentages_bal)]
labels = np.asarray(labels).reshape(2,2)
sb.heatmap(conf_mat_bal, annot=labels, fmt='', cmap='Blues')

# Print classification report with balanced weight
target_names = ['Credit not approved', 'Credit Approved']
print('Classification Report with balanced weight :')
print(classification_report(y_test, y_pred_bal, target_names=target_names))


# In[18]:


# Plot precision-recall curve for Logistic regression without balance weight
disp = plot_precision_recall_curve(logreg, X_test, y_test)
disp.ax_.set_title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision_score(y_test, y_pred)))


# In[19]:


# Plot precision-recall curve for Logistic regression with balance weight
disp = plot_precision_recall_curve(logreg_bal, X_test, y_test)
disp.ax_.set_title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision_score(y_test, y_pred_bal)))


# In[20]:


#Advanced Tasks


# In[21]:


# Apply logistic regression algorithm with Penalty L2
logreg_pen = LogisticRegression(penalty='l2') #By default, the penalty is L2
logreg_pen.fit(scaledX_train, y_train)
y_pred_pen = logreg_pen.predict(scaledX_test)

print('Sample Output of Prediction is:', y_pred_pen[0:4]) 

Decision_Function_pen = logreg_pen.decision_function(scaledX_test) 
print('Sample Output of L2 Penalty Decision Function is:', Decision_Function_pen[0:4])

pred_pro_pen = logreg_pen.predict_proba(scaledX_test)
print('Sample Output of  L2 Penalty Predict Proba Function is:', pred_pro_pen[0:4,:])

bal_accuracy_pen = balanced_accuracy_score(y_test, y_pred_pen)
print('L2 Penalty balance accuracy score: {:.4f}'.format(bal_accuracy_pen))


# Get the accuracy score of logreg model with Penalty L2 and print it
print("Accuracy of logistic regression classifier: ", logreg_pen.score(scaledX_test, y_test))


# Print the confusion matrix of the logreg model with Penalty L2
conf_mat_pen = confusion_matrix(y_pred_bal, y_test)

mat_names_pen = ['True Neg','False Pos','False Neg','True Pos']
mat_counts_pen = ['{0:0.0f}'.format(value) for value in
                conf_mat_pen.flatten()]
mat_percentages_pen = ['{0:.2%}'.format(value) for value in
                     conf_mat_pen.flatten()/np.sum(conf_mat_pen)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(mat_names_pen,mat_counts_pen,mat_percentages_pen)]
labels = np.asarray(labels).reshape(2,2)
sb.heatmap(conf_mat_pen, annot=labels, fmt='', cmap='Blues')

target_names = ['Credit not approved', 'Credit Approved']
print('Classification Report with L2 Penalty:')
print(classification_report(y_test, y_pred_pen, target_names=target_names))


# In[22]:


# Plot precision-recall curve for Logistic regression with penalty L2
disp = plot_precision_recall_curve(logreg_pen, X_test, y_test)


# In[23]:


# 2nd degree polynomial Expansion
poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_train)
print(X_poly.shape)
print(X_train.shape)

pipe = Pipeline([('polynomial_features',poly), ('logistic_regression',logreg)])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# In[ ]:




