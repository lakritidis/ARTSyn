import pandas as pd


class ResultHandler:
    def __init__(self, description, cv_results, out_path):
        self.description_ = description
        self.cv_results = cv_results
        self.mean_results = []
        self._out_path = out_path

    def to_df(self, result_type='folds'):
        # df_columns = ['Classifier', 'Sampler', 'Dataset',
        #              'Accuracy_Mean', 'Accuracy_Std', 'Balanced_Accuracy_Mean', 'Balanced_Accuracy_Std',
        #              'Sensitivity_Mean', 'Sensitivity_Std', 'Specificity_Mean', 'Specificity_Std',
        #              'F1_Mean', 'F1_Std', 'Precision_Mean', 'Precision_Std', 'Recall_Mean', 'Recall_Std',
        #              'Fit_Time', 'Order']

        df_columns = ['Dataset', 'Fold', 'Sampler', 'Classifier', 'Scorer', 'Val']
        df = pd.DataFrame(self.cv_results, columns=df_columns)

        if result_type != 'folds':
            df_mean_columns = ['Dataset', 'Sampler', 'Classifier', 'Scorer', 'Mean Val']

            classifiers = df['Classifier'].unique()
            samplers = df['Sampler'].unique()
            scorer = df['Scorer'].unique()
            folds = df['Fold'].unique()

            # return df_means

        return df

    def append(self, results, cv_results):
        for r in results:
            self.mean_results.append(r)

        for r in cv_results:
            self.cv_results.append(r)

    def to_latex(self, columns, orientation='vertical'):
        if orientation == 'vertical':
            df_temp = self.to_df(columns)
        else:
            df_temp = self.to_df(columns).transpose()
        latex_code = df_temp.to_latex()

        return latex_code

    def record_results(self):
        pd.set_option('display.precision', 3)

        # Results in latex format
        # f = open(self._base_path + filename + '_means.txt', 'w')
        # f.write(self.to_latex(columns=['Sampler', 'Accuracy_Mean'], orientation='horizontal'))
        # f.write(self.to_latex(columns=['Sampler', 'Balanced_Accuracy_Mean'], orientation='horizontal'))
        # f.write(self.to_latex(columns=['Sampler', 'Sensitivity_Mean'], orientation='horizontal'))
        # f.write(self.to_latex(columns=['Sampler', 'Specificity_Mean'], orientation='horizontal'))
        # f.write(self.to_latex(columns=['Sampler', 'F1_Mean'], orientation='horizontal'))
        # f.write(self.to_latex(columns=['Sampler', 'Precision_Mean'], orientation='horizontal'))
        # f.write(self.to_latex(columns=['Sampler', 'Recall_Mean'], orientation='horizontal'))

        # f.close()

        # Results in CSV format
        results = self.to_df().sort_values(by=['Dataset', 'Sampler', 'Scorer', 'Fold'])
        # print("self._out_path=", self._out_path)
        # print("self._description=", self.description_)
        results.to_csv(self._out_path + self.description_ + '.csv')
