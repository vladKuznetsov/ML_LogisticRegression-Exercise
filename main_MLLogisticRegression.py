from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score,recall_score, classification_report
from sklearn import preprocessing
from PandasInput import pandaUtils
import ModelTuning
from typing import List, Dict
from PipeLine_PCA_LogisticRegression import *

def fitAndTestModel(X_test, y_test, X_train, y_train, _features, _labels):

    accD = {}
    classMetricsD = {}
    reportD = {}
    for tol in [0.0001, 0.001, 0.01, 0.1]:

        model = buildModel(tol=tol)
        model.fit(X_train, y_train)

        accD[tol]  = [accuracy_score(y_test,   model.predict(X_test)) ,     \
                      accuracy_score(y_train,  model.predict(X_train)),   \
                      accuracy_score(_labels,   model.predict(_features))]

        t_precision = precision_score(y_test,  model.predict(X_test), average="weighted")
        t_recall    = recall_score   (y_test,  model.predict(X_test), average="weighted")

        classMetricsD[tol] = [t_precision, t_recall]

        reportD[tol] = classification_report(y_test, model.predict(X_test))
    return accD, classMetricsD, reportD

def printAccuracies(_accD : Dict[float, List[float]]):

    def printSliceOfTestScores(_accD: Dict[float, List[float]], _sliceId: int):
        for tol in _accD.keys():
            print("e1112 tol=%6.4f\ttest.accuracy  = %6.4f" % (tol, _accD[tol][_sliceId]))
        print ("\n\n")
        return

    print ("e1115 accuracies of test set")
    printSliceOfTestScores(_accD, 0)
    print ("e1116 accuracies of train set")
    printSliceOfTestScores(_accD, 1)
    print ("e1117 accuracies of test+train set")
    printSliceOfTestScores(_accD, 2)

    return

def printMetrics(_metricsD : Dict[float, List[float]]):

    print ("e1240 Average Precision/Recall by tol value.")

    for tol in _metricsD.keys():
        print("e1238 tol=%6.4f\tprecision  = %6.4f\trecall  = %6.4f" % (tol, _metricsD[tol][0], _metricsD[tol][1]))
    print ("\n\n")
    return

def printClassificationReports(_repD:Dict[float,str]):
    for tol in _repD.keys():
        print ("\ne2132 tol=%.4f \n" % tol)
        print (_repD[tol])

fileName = "/Volumes/DATA_1TB/Safary_Downloads/ml_data.csv"
features, labels = pandaUtils.readAndPrepareData(fileName, ['Unnamed: 0', 'ID'], ['mpr', 'nux'], 'outputs_class')

scaled_features =  preprocessing.scale(features)
# breaking data into train ant test sets using random sampling
X_test, X_train, y_test, y_train  = model_selection.train_test_split(scaled_features, labels, test_size=0.80, random_state=314)

lrm = buildModel()
lrm.fit(X_train, y_train)

y_pred = lrm.predict(X_test)

test_prec_score   = precision_score (y_test, y_pred, average="micro")
test_acc_score    = accuracy_score  (y_test, y_pred)
test_recall_score = recall_score    (y_test, y_pred, average="micro")

y_pred = lrm.predict(X_train)

train_prec_score   = precision_score(y_train, y_pred, average="micro")
train_acc_score    = accuracy_score (y_train, y_pred)
train_recall_score = recall_score   (y_train, y_pred, average="micro")

accD,metrD, repD = fitAndTestModel(X_test, y_test, X_train, y_train, scaled_features, labels)
printAccuracies(accD)
printMetrics(metrD)
printClassificationReports(repD)


param_grid ={   'PCA__n_components'           : [2,4,8,15], \
                'logical_regression__tol'     : [0.0001, 0.001, 0.01, 0.1], \
                'logical_regression__solver'  : ["newton-cg", "lbfgs", "liblinear"], \
                'logical_regression__multi_class'   : ["ovr", 'auto'],\
                'logical_regression__random_state'  : [0,314]}

if  check_param_grid(param_grid):
    pl = getPipeLine()
    ModelTuning.tunePipeLine(pl, param_grid, scaled_features, labels)
else:
    print ('e1309 Check your param_grid.')

pass
