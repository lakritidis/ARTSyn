import numpy as np
import pandas as pd
import gower

from scipy.stats import chi2_contingency

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

class TabularEvaluator:

    def __init__(self, df_real, df_syn, target, cat_idx, seed=42, n_splits=5):
        self.seed = seed
        self.n_splits = n_splits
        self.target = target

        self.df_real_raw = df_real.copy()
        self.df_syn_raw = df_syn.copy()

        self.cat_cols = list(df_real.columns[cat_idx])
        self.num_cols = [c for c in df_real.columns if c not in self.cat_cols ]

        self._validate_schema()

    ####################################################################
    # VALIDATION
    ####################################################################
    def _validate_schema(self):
        if list(self.df_real_raw.columns) != list(self.df_syn_raw.columns):
            raise ValueError("Real and synthetic datasets must have identical columns.")

    ####################################################################
    # PREPROCESSING
    ####################################################################
    def _prep_mixed(self, df):
        df = df.copy()

        for c in self.cat_cols:
            df[c] = df[c].astype("category")

        return df

    def _prep_gower(self, df):
        df = df.copy()

        for c in self.cat_cols:
            df[c] = df[c].astype(str)

        return df

    def _prep_ml(self, df):
        df = df.copy()

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X = pd.get_dummies(X, drop_first=True)

        return X, y

    ####################################################################
    # MIXED DEPENDENCY METRICS
    ####################################################################

    @staticmethod
    def cramers_v(x, y):
        confusion = pd.crosstab(x, y)
        if confusion.shape[0] < 2 or confusion.shape[1] < 2:
            return 0.0

        chi2 = chi2_contingency(confusion)[0]
        n = confusion.sum().sum()
        if n == 0:
            return 0.0

        r, k = confusion.shape
        return np.sqrt(chi2 / (n * max(min(k - 1, r - 1), 1)))

    @staticmethod
    def correlation_ratio(categories, measurements):
        categories = pd.Categorical(categories)

        if len(np.unique(measurements)) <= 1:
            return 0.0

        grand_mean = np.mean(measurements)
        ss_between = 0.0
        for cat in categories.categories:
            vals = measurements[categories == cat]
            if len(vals) == 0:
                continue

            ss_between += (len(vals) * (np.mean(vals) - grand_mean) ** 2)

        ss_total = np.sum((measurements - grand_mean) ** 2)

        if ss_total == 0:
            return 0.0

        eta2 = ss_between / ss_total
        return np.sqrt(eta2)

    def compute_mixed_matrix(self, df):
        cols = df.columns

        mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
        for col1 in cols:
            for col2 in cols:
                if col1 == col2:
                    mat.loc[col1, col2] = 1.0
                    continue

                ########################################################
                # numeric-numeric
                ########################################################
                if col1 in self.num_cols and col2 in self.num_cols:
                    val = df[col1].corr(df[col2])
                    if pd.isna(val):
                        val = 0.0

                    mat.loc[col1, col2] = val

                ########################################################
                # categorical-categorical
                ########################################################
                elif col1 in self.cat_cols and col2 in self.cat_cols:
                    mat.loc[col1, col2] = self.cramers_v(df[col1], df[col2])

                ########################################################
                # categorical-numeric
                ########################################################
                else:
                    if col1 in self.cat_cols:
                        cat = col1
                        num = col2
                    else:
                        cat = col2
                        num = col1

                    mat.loc[col1, col2] = (self.correlation_ratio(df[cat], df[num]))

        return mat

    def mixed_metrics(self):
        real = self._prep_mixed(self.df_real_raw)
        syn = self._prep_mixed(self.df_syn_raw)

        M_real = self.compute_mixed_matrix(real)
        M_syn = self.compute_mixed_matrix(syn)

        diff = np.abs(M_real - M_syn)

        diff = np.nan_to_num(diff)

        return { "mixed_mad": diff.values.mean(), "mixed_fro": np.linalg.norm(diff.values) }

    ####################################################################
    # GOWER + PRIVACY NN
    ####################################################################

    def gower_metrics(self):
        real = self._prep_gower(self.df_real_raw)
        syn = self._prep_gower(self.df_syn_raw)

        D = gower.gower_matrix(real, syn)
        D = np.nan_to_num(D)

        nn1 = D.min(axis=1)

        if D.shape[1] > 1:
            nn2 = np.partition(D, 1, axis=1)[:, 1]
        else:
            nn2 = nn1

        return {
            "gower_mean": D.mean(),
            "gower_min": D.min(),
            "nn1_mean": nn1.mean(),
            "nn1_min": nn1.min(),
            "nn_ratio": np.mean(nn1 / (nn2 + 1e-8))
        }

    ####################################################################
    # UTILITY
    ####################################################################
    def utility(self):
        Xr, yr = self._prep_ml(self.df_real_raw)
        Xs, ys = self._prep_ml(self.df_syn_raw)
        Xs = Xs.reindex(columns=Xr.columns, fill_value=0)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        rr_acc = []
        rr_f1 = []

        sr_acc = []
        sr_f1 = []

        for train_idx, test_idx in skf.split(Xr, yr):
            ########################################################
            # REAL -> REAL
            ########################################################
            X_train = Xr.iloc[train_idx]
            X_test = Xr.iloc[test_idx]

            y_train = yr.iloc[train_idx]
            y_test = yr.iloc[test_idx]

            model_r = XGBClassifier(eval_metric="logloss", random_state=self.seed)
            model_r.fit(X_train, y_train)
            pred_r = model_r.predict(X_test)

            rr_acc.append(accuracy_score(y_test, pred_r))
            rr_f1.append(f1_score(y_test, pred_r, average="weighted"))

            ########################################################
            # SYNTHETIC -> REAL
            ########################################################
            model_s = XGBClassifier(eval_metric="logloss", random_state=self.seed)
            model_s.fit(Xs, ys)
            pred_s = model_s.predict(X_test)

            sr_acc.append(accuracy_score(y_test, pred_s))
            sr_f1.append(f1_score(y_test, pred_s, average="weighted"))

        return {
            "rr_acc_mean": np.mean(rr_acc),
            "rr_acc_std": np.std(rr_acc),

            "rr_f1_mean": np.mean(rr_f1),
            "rr_f1_std": np.std(rr_f1),

            "sr_acc_mean": np.mean(sr_acc),
            "sr_acc_std": np.std(sr_acc),

            "sr_f1_mean": np.mean(sr_f1),
            "sr_f1_std": np.std(sr_f1),
        }

    ####################################################################
    # PRIVACY (DISTINGUISHABILITY)
    ####################################################################
    def privacy(self):
        df = pd.concat([self.df_real_raw, self.df_syn_raw], axis=0)
        y = np.array([1] * len(self.df_real_raw) + [0] * len(self.df_syn_raw))
        X = pd.get_dummies(df)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            clf = RandomForestClassifier(random_state=self.seed)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

            scores.append(accuracy_score(y_test, pred))

        return {
            "mia_mean": np.mean(scores),
            "mia_std": np.std(scores)
        }

    ####################################################################
    # MASTER EVALUATION
    ####################################################################
    def evaluate(self):
        results = {}
        results.update(self.mixed_metrics())
        results.update(self.gower_metrics())
        results.update(self.utility())
        results.update(self.privacy())

        return pd.Series(results)