from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class BaseClassifier:
    def __init__(self, name, short_name, model, **kwargs):
        self.name_ = name
        self.short_name_ = short_name
        self.model_ = model
        super().__init__(**kwargs)

    def fit(self, x, y):
        self.model_.fit(x, y)

    def predict(self, y):
        return self.model_.predict(y)


class Classifiers:
    def __init__(self, random_state=0):
        self.num_classifiers_ = 0

        self.models_ = (
            BaseClassifier(name="Logistic Regression", short_name="LR",
                           model=LogisticRegression(max_iter=300, random_state=random_state)),
            BaseClassifier(name="Support Vector Machine", short_name="SVM",
                           model=LinearSVC(C=1, random_state=random_state)),
            BaseClassifier(name="Decision Tree", short_name="DT",
                           model=DecisionTreeClassifier(criterion='gini', max_depth=None,
                                                        max_features=None, random_state=random_state)),
            BaseClassifier(name="XGBoost", short_name="XGBoost", model=xgb.XGBClassifier()),
            BaseClassifier(name="Random Forest", short_name="RF",
                           model=RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None,
                                                        max_features='sqrt', n_jobs=1, random_state=random_state)),
            BaseClassifier(name="Multilayer Perceptron", short_name="MLP",
                           model=MLPClassifier(activation='relu', hidden_layer_sizes=(128, 128), solver='adam',
                                               max_iter=300, random_state=random_state))
        )

        self.num_classifiers_ = len(self.models_)
