import timeit
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import warnings
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
warnings.filterwarnings("ignore")


#Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support

import streamlit as st

st.title('Deep Sight Analytics: Credit Card Fraud Detection!')

st.markdown('''
This is a **Credit Card Fraud Detection Dashboard** created in Streamlit.

''')


if st.sidebar.checkbox('Upload your CSV data'):
    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    credit_card_data = load_csv()
else:
    @st.cache
    def load_csv():
        credit_card_data=pd.read_csv('/Users/rohithvarma/Downloads/creditcard.csv')
        return credit_card_data
    credit_card_data = load_csv()

credit_card_data=st.experimental_data_editor(credit_card_data, num_rows="dynamic")

# Print the description and shape of the data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(credit_card_data.head(100))
    st.write('Shape of the dataframe: ',credit_card_data.shape)
    st.write('Data decription: \n',credit_card_data.describe())
        
# Print the number of valid transactions
fraudtransactions=credit_card_data[credit_card_data.Class==1]
realtransactions=credit_card_data[credit_card_data.Class==0]
outlier_percentage=(credit_card_data.Class.value_counts()[1]/credit_card_data.Class.value_counts()[0])*100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Percentage of Fraudulant transactions in the dataset: %.3f%%'%outlier_percentage)
    st.write('Number of Fraud Cases: ',len(fraudtransactions))
    st.write('Number of Valid Cases: ',len(realtransactions))
    
pr = ProfileReport(credit_card_data, explorative=True, minimal=True)
st.header('**Input DataFrame**')
st.write(credit_card_data)
st.write('---')
st_profile_report(pr)
   
#Obtaining the labels and features
X=credit_card_data.drop(['Class'], axis=1)
y=credit_card_data.Class

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 20)

#Print shape of train and test sets
if st.sidebar.checkbox('Show the Training and test data set'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)
    
logreg=LogisticRegression(random_state=42)
svm=SVC(random_state=42)
knn=KNeighborsClassifier()
rforest=RandomForestClassifier(random_state=42)
etree=ExtraTreesClassifier(random_state=42)
features=X_train.columns.tolist()


#Finding the most important features through feature importance function 
@st.cache
def feature_sort(model,X_train,y_train):
    
    mod=model    
    mod.fit(X_train, y_train)   
    imp = mod.feature_importances_
    return imp

#Classifiers for the feature importance function
trees=['Extra Trees','Random Forest']
feature_m = st.sidebar.selectbox('Which model for feature importance?', trees)

start_time = timeit.default_timer()
if feature_m=='Extra Trees':
    model=etree
    importance=feature_sort(model,X_train,y_train)
elif feature_m=='Random Forest':
    model=rforest
    importance=feature_sort(model,X_train,y_train)
elapsed_time = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes'%(elapsed_time/60))    

#Plot of feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()
from typing import List, Tuple

features: List[str] = features
importance: List[float] = importance

feature_importance: List[Tuple[str, float]] = list(zip(features, importance))

from operator import itemgetter
feature_sort = sorted(feature_importance, key=itemgetter(1), reverse=False)

#feature_sort=sorted(feature_importance, key = lambda x: x[1])


n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

top_features = list(list(zip(*feature_sort[:n_top_features]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=X_train_sfs
X_test_sfs_scaled=X_test_sfs



#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) 

smt = SMOTE()
nr = NearMiss()

def compute_performance(model, X_train, y_train,X_test,y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    Report=precision_recall_fscore_support(y_test, y_pred)
    "Precision: ", Report[0][0]
    "Recall: ", Report[1][0]
    "Fscore: ", Report[2][0]
    "Support: ", Report[3][0]
    elapsed_time = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes'%(elapsed_time/60)
    

          
if st.sidebar.checkbox('Run a credit card fraud detection model'):    
    alg_to_model = {
        'Extra Trees': etree,
        'Random Forest': rforest,
        'k Nearest Neighbor': knn,
        'Support Vector Machine': svm,
        'Logistic Regression': logreg
    }
    classifier = st.sidebar.selectbox('Which algorithm?', list(alg_to_model.keys()))
    rectifier = st.sidebar.selectbox('Which imbalanced class rectifier?', ['SMOTE', 'Near Miss', 'No Rectifier'])

    model = alg_to_model[classifier]

    if rectifier == 'No Rectifier':
        X_train_bal, y_train_bal = X_train_sfs_scaled, y_train
        compute_performance(model, X_train_bal, y_train_bal, X_test_sfs_scaled, y_test)
    elif rectifier == 'SMOTE':
        rect = smt
        X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
        st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
        X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
        st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
        compute_performance(model, X_train_bal, y_train_bal, X_test_sfs_scaled, y_test)
    elif rectifier == 'Near Miss':
        rect = nr
        X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
        st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
        X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
        st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
        compute_performance(model, X_train_bal, y_train_bal, X_test_sfs_scaled, y_test)


from tpot import TPOTClassifier
if st.sidebar.checkbox('Using TPOT Classifier'):
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=10,
                              offspring_size=None, mutation_rate=0.9,
                              crossover_rate=0.1,
                              scoring='accuracy', cv=5,
                              subsample=1.0, n_jobs=1,
                              max_time_mins=None, max_eval_time_mins=5,
                              random_state=None, config_dict=None,
                              warm_start=False,
                              memory=None,
                              periodic_checkpoint_folder=None,
                              early_stop=None,
                              verbosity=3,
                              disable_update_check=False)
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, use_dask=False)
    tpot.fit(X_train_sfs_scaled, y_train)
    tpot.score(X_test_sfs_scaled, y_test)   
    st.write('Results with TPOT %s'%tpot.score(X_test_sfs_scaled, y_test))







   

            
    
    
        


