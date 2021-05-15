from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from src.majority_voting import *
from src.evaluation_methods import *
from training.models.grid_search_cv import *
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

if __name__ == '__main__':
    df = pd.read_excel('../../data_based/excel/datasetV2.xlsx', sheet_name='Casos Dengue')

    x, y = df.iloc[:, :-1].values, df.iloc[:, -1:].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

    #under_sampler = RandomUnderSampler()
    #x_train, y_train = under_sampler.fit_resample(x_train, y_train)

    clf1 = LogisticRegression(penalty='l2',
                              C=0.001,
                              solver='lbfgs',
                              random_state=1,
                              class_weight="balanced",
                              multi_class="multinomial")

    clf2 = DecisionTreeClassifier(max_depth=1,
                                  criterion='entropy',
                                  random_state=0,
                                  class_weight="balanced")

    clf3 = KNeighborsClassifier(n_neighbors=1,
                                p=2,
                                metric='minkowski')

    pipe1 = Pipeline([['sc', StandardScaler()],
                      ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()],
                      ['clf', clf3]])

    clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

    clf_labels += ['Majority voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]

    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    rocAUC(x_train, y_train, x_test, y_test, all_clf, clf_labels, colors, linestyles)

    # buscar mejor modelo
    params = {'decisiontreeclassifier__max_depth': [1, 2],
              'pipeline-1__clf__C': [0.001, 0.1, 100.0]}

    model = GridSearchCV(estimator=mv_clf,
                        param_grid=params,
                        cv=10,
                        scoring='roc_auc_ovr')
    model = fit(model, x_train, y_train)

    mv_clf = bestEstimator(model)
    mv_clf = fit(mv_clf, x_train, y_train)
    y_pred = predict(mv_clf, x_test)

    print('Test Accuracy: %.3f' % score(mv_clf, x_test, y_test))
    confusionMatrix(y_test, y_pred)
