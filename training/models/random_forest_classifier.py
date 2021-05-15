from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt

def createModel(n_estimators, random_state):
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    return random_forest

def fit(model, x_train, y_train):
    model = clone(model).fit(x_train, y_train)
    variables = model.feature_importances_
    return model, variables

def predict(model, x_test):
    result = model.predict(x_test)
    return result

def showImportantColumns(columns, variables):
    columns_len = len(columns)

    indices = np.argsort(variables)[::-1]

    for f in range(columns_len):
        print("%2d) %-*s %f" % (f + 1, 30, columns[indices[f]], variables[indices[f]]), indices[f])

    plt.title('Feature Importance')
    plt.bar(range(columns_len), variables[indices], align='center')

    plt.xticks(range(columns_len), columns[indices], rotation=90)
    plt.xlim([-1, columns_len])
    plt.tight_layout()
    # plt.savefig('images/04_09.png', dpi=300)
    plt.show()