from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def makePipeLineModel(standar_scaler = None, dimension_reduction =None, model = LogisticRegression):
    pipe_lr = make_pipeline(standar_scaler, dimension_reduction, model)
    return pipe_lr

def fit(model, x_train, y_train):
    model = clone(model).fit(x_train, y_train)
    return model

def predict(model, x_test):
    result = model.predict(x_test)
    return result

def score(model, x_test, y_test):
    result = model.score(x_test, y_test)
    return result
