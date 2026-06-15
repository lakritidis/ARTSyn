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
                           model=LogisticRegression(C=1.0, l1_ratio=0.0, dual=False, tol=0.0001, fit_intercept=True,
                                                    intercept_scaling=1, max_iter=300, random_state=random_state)),

            BaseClassifier(name="Support Vector Machine", short_name="SVM",
                           model=LinearSVC(penalty='l2', loss='squared_hinge', tol=0.0001, C=1.0, random_state=random_state)),

            BaseClassifier(name="Decision Tree", short_name="DT",
                           model=DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                                        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                                        max_features=None, random_state=random_state)),

            BaseClassifier(name="XGBoost", short_name="XGBoost", model=xgb.XGBClassifier(random_state=random_state)),

            BaseClassifier(name="Random Forest", short_name="RF",
                           model=RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None,
                                                        max_features='sqrt', n_jobs=1, random_state=random_state)),

            BaseClassifier(name="Multilayer Perceptron", short_name="MLP",
                           model=MLPClassifier(activation='relu', hidden_layer_sizes=(128, 128), solver='adam',
                                               max_iter=300, random_state=random_state))
        )

        self.num_classifiers_ = len(self.models_)
