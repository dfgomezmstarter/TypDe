from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from src.evaluation_methods import *
from imblearn.under_sampling import RandomUnderSampler

if __name__ == '__main__':
    df = pd.read_excel('../../data_base/excel/datasetV2.xlsx', sheet_name='Casos Dengue')

    x, y = df.iloc[:, :-1].values, df.iloc[:, -1:].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

    under_sampler = RandomUnderSampler()
    x_train, y_train = under_sampler.fit_resample(x_train, y_train)
    #us = NearMiss(n_neighbors=3, version=2)
    #X_train_res, y_train_res = us.fit_sample(x_train, y_train)

    print("Distribution before resampling {}".format(y_train.shape))

    bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                    sampling_strategy='auto',
                                    replacement=False,
                                    random_state=0)

    # Train the classifier.
    bbc.fit(x_train, y_train)
    pred_y = bbc.predict(x_test)
    y_predict = bbc.predict(x_test)
    print('Test Accuracy: %.3f' % bbc.score(x_test, y_test))
    confusionMatrix(y_test, y_predict)
