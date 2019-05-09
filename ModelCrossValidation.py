from sklearn.model_selection import cross_val_score

def computeCrossValidation (model, test_data, test_target):

    scores = cross_val_score(model, test_data, test_target, cv=6, scoring='balanced_accuracy')
    print("e1132 Cross validation scores=",scores)
    print("e1606 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return