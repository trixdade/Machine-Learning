import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
%matplotlib inline
iris_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/train_iris.csv')
iris_data.drop(['Unnamed: 0'], axis=1, inplace=True)

X_train = iris_data.drop(['species'], axis=1)
y_train = iris_data.species
iris_data_test = pd.read_csv("https://stepik.org/media/attachments/course/4852/test_iris.csv")
iris_data_test.drop(['Unnamed: 0'], axis=1, inplace=True)
X_test = iris_data_test.drop(['species'], axis=1)
y_test = iris_data_test.species
np.random.seed(0)
scores_data = pd.DataFrame()
for max_depth in range(1, 100):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                    'train_score': [train_score],
                                    'test_score': [test_score]})
    scores_data = scores_data.append(temp_score_data)
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score'],
                           var_name='set_type', value_name='set_score')
sns.set_style("darkgrid")
sns.lineplot(x='max_depth', y='set_score', hue='set_type', data=scores_data_long)