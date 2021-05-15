from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from training.models.logistic_regression_pipeline import *
from src.evaluation_methods import *
from imblearn.under_sampling import RandomUnderSampler

if __name__ == '__main__':
    df = pd.read_excel('../../data_based/excel/datasetV2.xlsx', sheet_name='Casos Dengue')

    x, y = df.iloc[:, :-1].values, df.iloc[:, -1:].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

    under_sampler = RandomUnderSampler()
    x_train, y_train = under_sampler.fit_resample(x_train, y_train)

    print("Distribution before resampling {}".format(y_train.shape))

    clf2 = DecisionTreeClassifier(max_depth=2,
                                  criterion='entropy',
                                  random_state=0,
                                  class_weight="balanced")

    # Train the classifier.
    clf2.fit(x_train, y_train)
    pred_y = clf2.predict(x_test)
    y_predict = clf2.predict(x_test)
    print('Test Accuracy: %.3f' % clf2.score(x_test, y_test))
    confusionMatrix(y_test, y_predict)
