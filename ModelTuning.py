from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#import FeatureReductionPCA

def tunePipeLine(pl, _param_grid, train_data, train_target):

    clf = GridSearchCV(pl, _param_grid, scoring='balanced_accuracy', fit_params=None,
                       refit=True, cv=6, verbose=0)
    clf.fit(train_data, train_target)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r "  % (mean, std * 2, params))

    print("Detailed classification report:")
    print()
    print("Results for the model the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = train_target, clf.predict(train_data)
    print(classification_report(y_true, y_pred))
    print()

    return
