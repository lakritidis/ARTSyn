import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
import gower
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

from xgboost import XGBClassifier

class TabularEvaluator:
    def __init__(self, df_real, df_syn, target, cat_idx, seed=42):
        self.seed = seed
        self.target = target

        # Copy raw data (never modify originals)
        self.df_real_raw = df_real.copy()
        self.df_syn_raw  = df_syn.copy()

        self.cat_cols = df_real.columns[cat_idx]
        self.num_cols = [c for c in df_real.columns if c not in self.cat_cols]

        self._validate_schema()

    def _validate_schema(self):
        assert list(self.df_real_raw.columns) == list(self.df_syn_raw.columns), \
            "Real and synthetic must have identical columns"

    def _prep_mixed(self, df):
        df = df.copy()
        df[self.cat_cols] = df[self.cat_cols].astype("category")
        return df

    def _prep_gower(self, df):
        df = df.copy()
        df[self.cat_cols] = df[self.cat_cols].astype(str)
        return df

    def _prep_ml(self, df):
        df = df.copy()
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X = pd.get_dummies(X, drop_first=True)

        return X, y

    def mixed_matrix(self):
        df_r = self._prep_mixed(self.df_real_raw)
        df_s = self._prep_mixed(self.df_syn_raw)

        m_r = compute_mixed_matrix(df_r, self.cat_cols)
        m_s = compute_mixed_matrix(df_s, self.cat_cols)

        mad = np.mean(np.abs(m_r - m_s))

        return {
            "matrix_real": m_r,
            "matrix_syn": m_s,
            "mad": mad
        }

    def gower_metrics(self):
        df_r = self._prep_gower(self.df_real_raw)
        df_s = self._prep_gower(self.df_syn_raw)

        dist = gower.gower_matrix(df_r, df_s)

        nn1 = dist.min(axis=1)
        nn2 = np.partition(dist, 1, axis=1)[:, 1]

        return {
            "gower_mean": dist.mean(),
            "nn1_mean": nn1.mean(),
            "nn_ratio": (nn1 / (nn2 + 1e-8)).mean()
        }

    def privacy(self, n_splits=5):
        df = pd.concat([self.df_real_raw, self.df_syn_raw])
        y = np.array([1] * len(self.df_real_raw) + [0] * len(self.df_syn_raw))

        X = pd.get_dummies(df)

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.seed
        )

        scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = XGBClassifier(eval_metric="logloss", random_state=self.seed)
            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            scores.append(accuracy_score(y_test, preds))

        return {
            "mia_mean": np.mean(scores),
            "mia_std": np.std(scores)
        }

    def utility(self, n_splits=5):
        Xr, yr = self._prep_ml(self.df_real_raw)
        Xs, ys = self._prep_ml(self.df_syn_raw)

        # Align synthetic features to real
        Xs = Xs.reindex(columns=Xr.columns, fill_value=0)

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.seed
        )

        rr_scores = []
        sr_scores = []

        for train_idx, test_idx in skf.split(Xr, yr):
            X_train_r, X_test_r = Xr.iloc[train_idx], Xr.iloc[test_idx]
            y_train_r, y_test_r = yr.iloc[train_idx], yr.iloc[test_idx]

            # --- Real → Real ---
            model_r = XGBClassifier(
                eval_metric="logloss",
                random_state=self.seed
            )
            model_r.fit(X_train_r, y_train_r)
            preds_r = model_r.predict(X_test_r)
            rr_scores.append(accuracy_score(y_test_r, preds_r))

            # --- Synthetic → Real (TSTR) ---
            model_s = XGBClassifier(
                eval_metric="logloss",
                random_state=self.seed
            )
            model_s.fit(Xs, ys)  # train ONCE on full synthetic
            preds_s = model_s.predict(X_test_r)
            sr_scores.append(accuracy_score(y_test_r, preds_s))

        return {
            "acc_rr_mean": np.mean(rr_scores),
            "acc_rr_std": np.std(rr_scores),
            "acc_sr_mean": np.mean(sr_scores),
            "acc_sr_std": np.std(sr_scores),
        }

    def evaluate(self):
        results = {}

        results.update(self.mixed_matrix())
        results.update(self.gower_metrics())
        results.update(self.utility())
        results.update(self.privacy())

        return results

def run_multiple(df_real, df_syn, target, cat_idx, n_runs=10):
    results = []

    for seed in range(n_runs):
        evaluator = TabularEvaluator(
            df_real, df_syn, target, cat_idx, seed
        )
        results.append(evaluator.evaluate())

    return pd.DataFrame(results)


# Helper Functions
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1) + 1e-8)))

def correlation_ratio(categories, measurements):
    categories = pd.Categorical(categories)
    groups = [measurements[categories == cat] for cat in categories.categories]

    grand_mean = np.mean(measurements)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((measurements - grand_mean)**2)

    return np.sqrt(ss_between / (ss_total + 1e-8))

def compute_mixed_matrix(df, cat_cols):
    cols = df.columns
    mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    num_cols = [c for c in cols if c not in cat_cols]

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):

            if col1 == col2:
                mat.loc[col1, col2] = 1.0

            elif col1 in num_cols and col2 in num_cols:
                mat.loc[col1, col2] = df[col1].corr(df[col2])

            elif col1 in cat_cols and col2 in cat_cols:
                mat.loc[col1, col2] = cramers_v(df[col1], df[col2])

            else:
                # numeric-categorical
                if col1 in cat_cols:
                    cat, num = col1, col2
                else:
                    cat, num = col2, col1

                mat.loc[col1, col2] = correlation_ratio(df[cat], df[num])

    return mat

def summarize_metric(df, col):
    mean = df[col].mean()
    std = df[col].std()
    ci = 1.96 * std / np.sqrt(len(df))
    return mean, (mean - ci, mean + ci)