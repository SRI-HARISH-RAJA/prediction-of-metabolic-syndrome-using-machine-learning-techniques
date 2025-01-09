#Import necessary libraries

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from timeit import default_timer as timer
from sklearn import preprocessing
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder  

data = pd.read_csv(r"/content/Metabolic Syndrome.csv")

print(data.info())
data.shape
data.isnull().sum()
data_filtered = data[data.isnull().any(axis=1)]
data_filtered
data["Marital"].unique()
data = data.dropna()
data.isnull().sum()

int_vars = data.select_dtypes(include=['int', 'float'])

fig = make_subplots(rows=6, cols=2, subplot_titles=int_vars.columns)

for i, var in enumerate(int_vars.columns):
    row = i // 2 + 1
    col = i % 2 + 1
    trace = go.Histogram(x=data[var], nbinsx=30, name=var)
    fig.add_trace(trace, row=row, col=col)

fig.update_layout(
    title_text="Histograms for Numeric Variables",
    height=1100,
    width=900
)
fig.show()

data.head()

int_vars = ['Triglycerides', 'Age', 'Race']

fig = make_subplots(rows=2, cols=2, subplot_titles=int_vars, shared_xaxes=True)

for i, var in enumerate(int_vars):
    row = (i // 2) + 1
    col = (i % 2) + 1
    trace = go.Histogram(x=data[var], name=var)
    fig.add_trace(trace, row=row, col=col)

fig.update_layout(
    title_text="Count Plots for Selected Numeric Variables",
    showlegend=False,
    height=600,
    width=900
)
fig.show()

fig = px.histogram(data, x='MetabolicSyndrome', color='MetabolicSyndrome', title='Count Plot for MetabolicSyndrome')

fig.update_layout(
    showlegend=False,
    xaxis_title='MetabolicSyndrome',
    yaxis_title='Count',
    height=500,
    width=600
)
fig.show()

int_vars = data.select_dtypes(include=['int', 'float']).columns

fig = make_subplots(rows=4, cols=3, subplot_titles=int_vars)

for i, var in enumerate(int_vars):
    row = i // 3 + 1
    col = i % 3 + 1
    trace = go.Box(x=data[var], name=var)
    fig.add_trace(trace, row=row, col=col)

fig.update_layout(
    title_text="Boxplots for Selected Numeric Variables (Outliers Detection)",
    showlegend=False,
    height=1200,
    width=1000
)
fig.show()

Features_with_outliers=['WaistCirc', 'BMI', 'UrAlbCr', 'UricAcid','BloodGlucose','HDL','Triglycerides']

def remove_outliers_iqr(data):

    # Calculating the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculating the interquartile range (IQR)
    IQR = Q3 - Q1

    # Defining the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Removing outliers
    data = np.where(data>upper_bound, upper_bound, np.where(data<lower_bound,lower_bound,data))

    return data[(data >= lower_bound) & (data <= upper_bound)]

for column in Features_with_outliers:
  data[column] = remove_outliers_iqr(data[column])

#outliers detection after removing them
int_vars = data.select_dtypes(include = ['int','float'])

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15,10))
axs = axs.flatten()

for i, var in enumerate (int_vars):
    sns.boxplot(x=var,data=data,ax=axs[i])
    axs[i].set_title(var)

plt.tight_layout()
plt.show()

label_encoders = {} # Create an empty dictionary to store the label encoders
for col in data.columns:
    if data[col].dtype == 'object':
        le = preprocessing.LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
X=data.drop('MetabolicSyndrome',axis=1)
y=data['MetabolicSyndrome']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 49)

def report(y_test, pred):
    acc = accuracy_score(y_test, pred)
    error_rate = 1 - acc
    cm = confusion_matrix(y_test, pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    precision = tp/(tp+fp)
    sensitivity = tp/(tp/fn)
    specificity = tn/(tn+fp)
    recall = tp/(tp+fn)
    f1_score = 2*((precision*recall)/(precision+recall))
    print('\nConfusion matrix : \n',cm)
    print('\nAccuracy Score : ',round(acc,4))
    print('\nError rate : ',round(error_rate,4))
    print('\nPrecision : ',round(precision,4))
    print('\nRecall : ',round(recall,4))
    print('\nF1-Score : ',round(f1_score,4))
    print('\nSensitivity : ',round(sensitivity,4))
    print('\nSpecificity : ',round(specificity,4))

    start = timer()
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
end = timer()
print('Random Forest : ')
print('Computional time for the Random Forest is (in sec) : ',(end - start))
report(y_test, rf_pred)


start = timer()
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
end = timer()
print('\n\nK Nearest Neighbor : ')
print('Computional time for the K Nearest Neigbor is (in sec) : ',(end - start))
report(y_test, knn_pred)

start = timer()
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
end = timer()
print('\n\nLogistic Regression : ')
print('Computional time for the Logistic Regression is (in sec) : ',(end - start))
report(y_test, lr_pred)


!pip install scikit-learn
from sklearn.metrics import roc_curve, auc
# Assuming you have x_train, y_train, x_test, and y_test and have preprocessed the data appropriately

# Initialize and train the models
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

lr = LogisticRegression()
lr.fit(x_train, y_train)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Get predicted probabilities for the positive class
rf_probs = rf.predict_proba(x_test)[:, 1]
lr_probs = lr.predict_proba(x_test)[:, 1]
knn_probs = knn.predict_proba(x_test)[:, 1]

# Calculate ROC curve and AUC for Random Forest
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_roc_auc = auc(rf_fpr, rf_tpr)

# Calculate ROC curve and AUC for Logistic Regression
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
lr_roc_auc = auc(lr_fpr, lr_tpr)

# Calculate ROC curve and AUC for KNN
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
knn_roc_auc = auc(knn_fpr, knn_tpr)

# Plot ROC curves
plt.figure()
plt.plot(rf_fpr, rf_tpr, color='darkorange', lw=2, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot(lr_fpr, lr_tpr, color='green', lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
plt.plot(knn_fpr, knn_tpr, color='blue', lw=2, label='KNN (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the models
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Get predicted probabilities for the positive class
rf_probs = rf.predict_proba(X_test)[:, 1]
lr_probs = lr.predict_proba(X_test)[:, 1]
knn_probs = knn.predict_proba(X_test)[:, 1]

# Calculate PR curve and AUC for Random Forest
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
rf_pr_auc = auc(rf_recall, rf_precision)

# Calculate PR curve and AUC for Logistic Regression
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_pr_auc = auc(lr_recall, lr_precision)

# Calculate PR curve and AUC for KNN
knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn_probs)
knn_pr_auc = auc(knn_recall, knn_precision)

# Plot PR curves
plt.figure()
plt.plot(rf_recall, rf_precision, color='darkorange', lw=2, label='Random Forest (area = %0.2f)' % rf_pr_auc)
plt.plot(lr_recall, lr_precision, color='green', lw=2, label='Logistic Regression (area = %0.2f)' % lr_pr_auc)
plt.plot(knn_recall, knn_precision, color='blue', lw=2, label='KNN (area = %0.2f)' % knn_pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the models
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Get predicted probabilities for the positive class
rf_probs = rf.predict_proba(X_test)[:, 1]
lr_probs = lr.predict_proba(X_test)[:, 1]
knn_probs = knn.predict_proba(X_test)[:, 1]

# Calculate Brier score for Random Forest
rf_brier = brier_score_loss(y_test, rf_probs)

# Calculate Brier score for Logistic Regression
lr_brier = brier_score_loss(y_test, lr_probs)

# Calculate Brier score for KNN
knn_brier = brier_score_loss(y_test, knn_probs)

# Print the Brier scores
print('Random Forest Brier score:', rf_brier)
print('Logistic Regression Brier score:', lr_brier)
print('KNN Brier score:', knn_brier)

