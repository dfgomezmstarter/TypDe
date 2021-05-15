from sklearn.model_selection import train_test_split
import pandas as pd
from training.models.random_forest_classifier import *

if __name__ == '__main__':
    df = pd.read_excel('../../data_based/excel/datasetV2.xlsx', sheet_name='Casos Dengue')

    # Separación de los datos
    x, y = df.iloc[:, :-1].values, df.iloc[:, -1:].values
    x_train, y_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    feat_labels = df.columns[:-1]

    forest = createModel(500, 1)
    forest, variables = fit(forest, x_train, y_train)
    showImportantColumns(feat_labels, variables)
