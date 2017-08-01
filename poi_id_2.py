#!/usr/bin/python


import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from sklearn.grid_search import GridSearchCV

from tester_2 import dump_classifier_and_data
from tester_2 import test_classifier



import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

from enron_2 import cool_print, calc_percent_missing, tune_and_eval_clf

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
poi_label = 'poi'
email_features = ['from_messages', 
                  'to_messages',
                  'from_poi_to_this_person', 
                  'from_this_person_to_poi', 
                  'shared_receipt_with_poi']
financial_features = ['salary', 
                      'bonus', 
                      'total_payments',
                      'total_stock_value',
                      'deferral_payments', 
                      'deferred_income', 
                      'director_fees', 
                      'exercised_stock_options', 
                      'expenses', 
                      'loan_advances', 
                      'long_term_incentive', 
                      'restricted_stock', 
                      'restricted_stock_deferred',
                      'other']


features_list = [poi_label] + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Store to my_dataset for easy export below.
my_dataset = data_dict


#We will use pandas Dataframe to ease Data Exploration and Preprocessing

# 1. replace NaN text strings
for name, d in my_dataset.items():
    for feature, value in d.items():
        if value == 'NaN':
            d[feature] = np.nan

# 2. convert of dictionaries into pandas.DataFrame
df = pd.concat(objs=[pd.DataFrame.from_dict(data={k:d}, orient='index') for k, d in my_dataset.items()])

#################################################
# DATA OVERVIEW AND IDENTIFICATION OF OUTLIERS  #
###############################################
cool_print('DATASET INFO')
print df.info()

### Task 2: Remove outliers

#Add new column just for identification of percentage of missing values for the outliers
calc_percent_missing(df)

#Save dataframe to excel for manual outliers identification
#df.to_excel('my_dataframe.xlsx') 

# Plot scatterplot to find outliers
grid = sns.JointGrid(df['salary'], df['total_stock_value'], space=0, size=8, ratio=70)
grid.plot_joint(plt.scatter, color="b")
grid.plot_marginals(sns.rugplot, height=1, color="b")
plt.show()

#Print top total_stock_value records
cool_print('TOP TOTAL STOCK VALUE')

print df['total_stock_value'].sort_values(ascending=False).head()

#Print records with more missing values
cool_print('TOP MISSING VALUES RECORDS')
print df['percent_missing'].sort_values(ascending=False).head()

#Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']

df = df.drop(outliers)

print '-> Shape after outliers removal: ', df.shape

# Plot scatterplot to after finding outliers
grid = sns.JointGrid(df['salary'], df['total_stock_value'], space=0, size=8, ratio=70)
grid.plot_joint(plt.scatter, color="b")
grid.plot_marginals(sns.rugplot, height=1, color="b")
plt.show()

###################
# PRE-PROCESSING #
#################

### Task 3: Create new feature(s)
# messages_to_poi = from_this_person_to_poi/from_messages
df['messages_to_poi'] = df['from_this_person_to_poi']*1.0/df['from_messages']

# messages_from_poi = from_poi_to_this_person/to_messages
df['messages_from_poi'] = df['from_poi_to_this_person']*1.0/df['to_messages']

#updating the feature list
my_feature_list = features_list + ['messages_to_poi', 'messages_from_poi']

#Drop columns with too many missing values
features_to_delete = ['deferral_payments', 'restricted_stock_deferred',
                     'loan_advances', 'director_fees']

df = df.drop(labels=features_to_delete, axis=1)

cool_print('DATAFRAME AFTER FEATURES UPDATE')
print df.info()

for col in features_to_delete:
    my_feature_list.remove(col)

#Re-calculate percent missing
calc_percent_missing(df)

#Remove rows with too many missing values
max_percent_missing = 75.0

cool_print('ROWS TO BE REMOVED WITH TOO MANY MISSING VALUES')
print df[df['percent_missing'] > max_percent_missing].index

df = df[df['percent_missing'] <= max_percent_missing]

cool_print('DATAFRAME INFO AFTER CLEANING')
print df.info()

#Fill missing values with the mean of it´s own column

for feature in my_feature_list[1:]:
    df[feature].fillna(value=df[feature].mean(), inplace=True)
    
#Scale values
df[my_feature_list[1:]] = scale(df[my_feature_list[1:]].values)

#Select best k features
k_best = SelectKBest()
k_best.fit(df[my_feature_list[1:]].values, df['poi'])
scores = k_best.scores_
unsorted_pairs = zip(my_feature_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[-1])))

cool_print('BEST FEATURES')

for feature, score in sorted_pairs:
    print '{:4.2f} | {} |'.format(score, feature)



#Save to my_dataframe excel
df.to_excel('my_dataframe.xlsx') 

# Converting back from pandas.DataFrame to dictionary
#
#
my_dataset = {}
for row in df.itertuples():
    d = row._asdict()
    name = d.pop('Index')
    my_dataset[name] = d   

### For local testing labels and features are extracted
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

######################
# TRYING ALGORITHMS #
####################
### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



##Logistic Regression Classifier
cool_print('LOGISTIC REGRESSION')
lr_clf = LogisticRegression()
lr_params = {'clf__C': [0.1, 0.01, 0.001, 0.0001],
             'clf__tol': [1, 0.1, 0.01, 0.001, 0.0001],
             'clf__penalty': ['l1', 'l2'],
             'clf__random_state': [42, 46, 60]}

tune_and_eval_clf(lr_clf, lr_params, features, labels)

##Support Vector Machine Classifier
cool_print('SUPPORT VECTOR MACHINES')
svc_clf = SVC()
svc_params = {'clf__C': [1000], 'clf__gamma': [0.001], 'clf__kernel': ['rbf']}

tune_and_eval_clf(svc_clf, svc_params, features, labels)


##Decision Tree Classifier
cool_print('DECISION TREE')
dt_clf = DecisionTreeClassifier()
dt_params = {"clf__min_samples_leaf": [2, 6, 10, 12],
             "clf__min_samples_split": [2, 6, 10, 12],
             "clf__criterion": ["entropy", "gini"],
             "clf__max_depth": [None, 5],
             "clf__random_state": [42, 46, 60]}

tune_and_eval_clf(dt_clf, dt_params, features, labels)

#after exchanging with Myles I have used the test_classifier results for the evaluation
#for Question 3, as there are too less POI´s for the test reasons
test_classifier(lr_clf, my_dataset, my_feature_list)
test_classifier(svc_clf, my_dataset, my_feature_list)
test_classifier(dt_clf, my_dataset, my_feature_list)


# BEST ALGORITH = LOGISTIC REGRESSION
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
    
def select_and_dump(algorithm):
    """ Select and dump the algorithm selected. """
    
    algorithm = algorithm.strip().replace(' ','').lower()
    
    if algorithm == 'logisticregression': 
     
        my_clf = LogisticRegression(C=1e-08, penalty='l2', random_state=42, tol=0.01)
        number_best_features = 7
    
    elif algorithm in ['svc', 'svm', 'supportvectormachines']:
        my_clf = SVC(C=1000, gamma=0.001, kernel='rbf')
        number_best_features = 7
    
    elif algorithm == 'decisiontree':
        my_clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=2, min_samples_split=6,random_state=42)
        number_best_features = 7 
    
    else:
        raise ValueError('Algorithm must be LogisticRegression, SVC, DecisionTree')
      
    clf = my_clf
    best_feature_list = [x[0] for x in sorted_pairs[:number_best_features]]
    
    #My Feature List after selecting K Best featues
    my_feature_list = ['poi'] + best_feature_list 
    
    dump_classifier_and_data(clf, my_dataset, my_feature_list)


select_and_dump('logisticregression')
