import pickle


def fit(model, x_train, y_train):
    model = model.fit(x_train, y_train)
    return model

def predict(model, x_test):
    result = model.predict(x_test)
    return result

def bestEstimator(model):
    for r, _ in enumerate(model.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (model.cv_results_['mean_test_score'][r],
                 model.cv_results_['std_test_score'][r] / 2.0,
                 model.cv_results_['params'][r]))

    mv_clf = model.best_estimator_
    mv_clf.set_params(**model.best_estimator_.get_params())
    print(mv_clf)
    return mv_clf

def score(model, x_test, y_test):
    result = model.score(x_test, y_test)
    return result

def save_model(model):
    filename = '../load_model/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))