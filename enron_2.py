# -*- coding: utf-8 -*-
import sklearn.feature_selection
import sklearn.pipeline
import sklearn.model_selection
import sklearn.metrics

def cool_print(msg):
    """Prints out messages in cool way"""
    print '#'*(len(msg)+4)
    print '# {} #'.format(msg)
    print '#'*(len(msg)+4)
    
#
def calc_percent_missing(df):
    """Calculates percentage of missing columns for each record (row)."""
    for index, row in df.iterrows():
        df.ix[index, 'percent_missing'] = (row.size-row.count())*100.0/row.size    
#
# Function created inspired on:
#https://www.civisanalytics.com/blog/workflows-in-python-using-pipeline-and-gridsearchcv-for-more-compact-and-comprehensive-code/
#
def tune_and_eval_clf(clf, params, features, poi_labels):
    """ Tunes and evaluates a classifier """
    
    select = sklearn.feature_selection.SelectKBest()
        
#    print clf
    
    steps = [('feature_selection', select), ('clf', clf)]
    
    pipeline = sklearn.pipeline.Pipeline(steps)
    
    parameters = dict(feature_selection__k=[7, 10, 15])
    parameters.update(params)

    features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, poi_labels, test_size=0.3, random_state=42)
    
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=60)
    
    #to optimize Precision and Recall I have used the F1 score (suggestion of Myles)
    cv = sklearn.model_selection.GridSearchCV(estimator=pipeline, cv=sss, param_grid=parameters, scoring='f1')
   
    cv.fit(features_train, labels_train)
    
    print 'Parameters Tuned:'
    best_parameters = cv.best_params_
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    labels_pred = cv.predict(features_test)
    
    target_names = ['NON-POI', 'POI']
    
#    report = sklearn.metrics.classification_report( labels_test, labels_pred, target_names=target_names )
#    print 'Report:'
#    print report
    
    precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(labels_test, labels_pred, labels=target_names, average='binary')
    print 'Performance 1-Run:'
    print 'Precision = {:8.6f} and Recall = {:7.5f}'.format(precision, recall)
