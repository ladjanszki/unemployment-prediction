import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unidecode
from itertools import product
from datetime import datetime


import tsm_utils as tsm
import tsm_preprocessor as prep

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def dict_product(d):
    """
    Function for generating the cartesian product of a dict of lists
    """
    sorted_keys = sorted(d.keys())
    for element in product(*list(d[k] for k in sorted_keys)):
        yield dict(zip(sorted_keys, element))

def runModel(modelToRun, predictInput, nCycles, paramGrid, testRatio):
    """
    General function to run a scikit learn model
    """

    histogram = np.zeros(len(predictInput))
    I = np.ones(len(predictInput))
    normalizer = 0
    
    for i in range(0, nCycles):
        trainInput, validationInput, trainOutput, validationOutput = train_test_split(allInput, allOutput, test_size = testRatio)
        for j, element in enumerate(dict_product(paramGrid)):
    
            clf = modelToRun(**element)
            clf = clf.fit(trainInput, trainOutput) 
            score = clf.score(validationInput, validationOutput)
            out = clf.predict(predictInput)
    
            histogram += score * (I - out)
            normalizer += 1
    
            #TODO: print model, cycle, score on validation, number of unemployed
            print('Model ' + str(j) + ' in cycle ' + str(i) + ' Unemployed number: ' + str(out.tolist().count(0)) + ' Score: ' + str(score))
         
    # Postprocessing
    histogram = histogram / normalizer
    return histogram

def plotResults(x, y, fileName):
    """
    Function to plot the histogram
    """

    myDpi = 100
    xres = 1920
    yres = 981
    
    xsize = float(xres) / float(myDpi)
    ysize = float(yres) / float(myDpi)
    
    fig = plt.figure(figsize=(xsize,ysize))
    ax = plt.subplot(111)
    #ax = plt.subplot(1, 1, 1)

    #ax.bar(toPlot['plotlabel'], toPlot['svc_likelihood'])
    #ax.set_xticks(toPlot['plotlabel'])
    #ax.set_xticklabels(toPlot['plotlabel'], rotation=90) 

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90) 
    
    #plt.show()
    #fig.savefig('svc.png')
    fig.savefig(fileName)
 

# TODO: create a ,,best model fitter'' function from this
# Searching for the best classifier
#svcClf = GridSearchCV(SVC(), param_grid)
#svcClf = svcClf.fit(trainInput, trainOutput) 
##print("Best estimator found by grid search:")
##print(svcClf.best_estimator_)
#print(svcClf.score(validationInput, validationOutput))                           
#svcOut = svcClf.predict(predictInput)
#print(svcOut)
#print('Unemployed number: ', svcOut.tolist().count(0))


 
########################### Main script ############################
 
# Loading residents data
jarasok = tsm.preprocessingResidentData('jarasok_clean.csv') # Reading in resident numbers

# Data for learning and model validation
features = pd.read_csv('orig_data/train_verseny.csv')   # Reading into the DataFrame from csv
merged = prep.preprocessData(features, jarasok, 2)
merged = prep.preprocessDolg(merged)

# Data for prediction
predictRaw = pd.read_csv('orig_data/test_verseny.csv')
predictProcessed = prep.preprocessData(predictRaw, jarasok, 0)

# Setting Input and Output
#inputColumns = ['iskola', 'neme', 'eletpalya', 'lakossag', 'd10', 'd11', 'egyutt']
#inputColumns = ['iskola', 'eletpalya', 'lakossag', 'd10', 'd11']
#inputColumns = ['iskola', 'neme', 'eletpalya', 'lakossag', 'egyutt']
#inputColumns = ['iskola', 'eletpalya', 'lakossag', 'piping']
inputColumns = ['iskola', 'eletpalya', 'lakossag']
predictInput = predictProcessed[inputColumns]

allInput = merged[inputColumns]
allOutput = merged['dolg']

## New DataFrame for output for all model 
predictOutput = pd.DataFrame()
predictOutput['sorszam'] = predictProcessed['sorszam']
predictOutput['plotlabel'] = predictOutput['sorszam'].apply(str)
predictOutput['plotlabel'] = predictOutput['plotlabel'].astype(str)

# DEPRECATED
predictIndex = predictProcessed['sorszam']
predictIndex = predictIndex.astype(str)

# Support Vector Classifier ############################################
print('Support Vector Classifier: ')
svc_param_grid = {
    'C' : [0.01, 0.1, 1, 10, 100],  
    'cache_size' : [200], 
    'class_weight' : ['balanced'], 
    'coef0' : [0.0],
    'decision_function_shape' : ['ovr', 'ovo'], 
    'degree' : [3], 
    'gamma' : ['auto'], 
    'kernel' : ['rbf', 'linear'],
    'max_iter' : [-1], 
    'probability' : [False], 
    'random_state' : [None], 
    'shrinking' : [True],
    'tol' : [0.001], 
    'verbose' : [False]
}

# Running all the models and adding to output dataframe
histogram = runModel(SVC, predictInput, 2, svc_param_grid, 0.2)
predictOutput['svc_likelihood'] = histogram

# Neural Network #######################################################
print('Neural network: ')
nn_param_grid = {
        'hidden_layer_sizes' : [(12), (10), (8), (6),(5), (10, 5), (8, 5), (12, 10)],
        'max_iter' : [1000],
        'activation' : ['relu', 'logistic'], 
        #'alpha' : 1e-05, 
        #'batch_size' : 'auto', 
        #'beta_1' : 0.9,
        #'beta_2' : 0.999, 
        #'early_stopping' : False, 
        #'epsilon' : 1e-08,
        'learning_rate' : ['constant', 'adaptive'],
        'max_iter' : [100, 200, 500, 1000],
        #'learning_rate_init' : 0.001, 
        #'momentum' : 0.9,
        #'nesterovs_momentum' : True, 
        #'power_t' : 0.5, 
        #'random_state' : 1, 
        #'shuffle' : True,
        'solver' : ['adam', 'lbfgs'], 
        #'tol' : 0.0001, 
        #'validation_fraction' : 0.1, 
        #'verbose' : False,
        'warm_start' : [False, True]
}

histogram = runModel(MLPClassifier, predictInput, 3, nn_param_grid, 0.2)
predictOutput['nn_likelihood'] = histogram

# Random Forest Classifier #############################################
print('Random forest: ')
rf_param_grid = {
        'n_estimators': [1, 10, 100, 200, 500, 1000],
        'bootstrap': [True, False],
        'max_features': ['auto', None]
        }    

histogram = runModel(RandomForestClassifier, predictInput, 3, rf_param_grid, 0.2)
predictOutput['rf_likelihood'] = histogram

# Postprocessing the result ############################################
predictOutput['sum'] = predictOutput['rf_likelihood'] + predictOutput['nn_likelihood'] + predictOutput['svc_likelihood']
predictOutput.sort_values(by=['sum'], inplace = True, ascending=False)

# Saving ###############################################################
prefix = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
outPath = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
if not os.path.exists(outPath):
    print("Creating path: " + outPath)
    os.makedirs(outPath)
    
#outputFile =  prefix + '_out.csv'
#svcPlot = prefix + '_svc.png'
#nnPlot = prefix + '_nn.png'
#rfPlot = prefix + '_rf.png'
#sumPlot = prefix + '_sum.png'

outputFile =  outPath + '/out.csv'
svcPlot = outPath + '/svc.png'
nnPlot = outPath + '/nn.png'
rfPlot = outPath + '/rf.png'
sumPlot = outPath + '/sum.png'
   
# Saving the data
predictOutput.to_csv(outputFile)
print(predictOutput.head(30))

# Saving the plots
plotResults(predictOutput['plotlabel'], predictOutput['svc_likelihood'], svcPlot)
plotResults(predictOutput['plotlabel'], predictOutput['nn_likelihood'], nnPlot)
plotResults(predictOutput['plotlabel'], predictOutput['rf_likelihood'], rfPlot)
plotResults(predictOutput['plotlabel'], predictOutput['sum'], sumPlot)
 



