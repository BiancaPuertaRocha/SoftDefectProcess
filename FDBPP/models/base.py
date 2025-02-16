import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

class ModelEvaluationResult:
    """
    Stores the results of a machine learning model evaluation.

    Attributes:
        auc_score (float): The AUC (Area Under the ROC Curve) score of the evaluated model.
        y_test (array-like): The actual labels of the test set.
        y_pred (array-like): The model's predictions on the test set.
    """

    def __init__(self, auc_score: float, y_test, y_pred):
        """
        Initializes the ModelEvaluationResult with evaluation metrics.

        Args:
            auc_score (float): The computed AUC score.
            y_test (array-like): The actual labels of the test set.
            y_pred (array-like): The predicted labels by the model.
        """
        self.auc_score = auc_score
        self.y_test = y_test
        self.y_pred = y_pred

    def __repr__(self):
        return f"ModelEvaluationResult(auc_score={self.auc_score:.4f})"

    def to_dict(self):
        """
        Converts the evaluation results into a dictionary.

        Returns:
            dict: A dictionary containing auc_score, y_test, and y_pred.
        """
        return {
            "auc_score": self.auc_score,
            "y_test": self.y_test,
            "y_pred": self.y_pred
        }


class GenericModelBase:
    def prepare_data(self, df, sampler=None, feature_selector=None, threshold=0.5, k=10, balance_first=False):
        """Prepara os dados aplicando tratamento de nulos, balanceamento, seleção de atributos e normalização."""
        X = df.drop('failure_prone', axis=1)
        y = df['failure_prone']

        # Remover valores nulos
        X = X.dropna()
        y = y.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)

        # Remover colunas constantes
        X = X.loc[:, X.nunique() > 1]

        if balance_first and sampler:
            X, y = self.apply_sampler(X, y, sampler)

        if feature_selector:
            selected_features = feature_selector(X, y, k=k) if feature_selector.__name__ == 'chi_square_feature_selection' else feature_selector(X, y, threshold=threshold)
            X = X[selected_features]

        if not balance_first and sampler:
            X, y = self.apply_sampler(X, y, sampler)

        # Codificar variáveis categóricas
        X = pd.get_dummies(X, columns=X.select_dtypes(include=['object', 'category']).columns)

        # Normalização
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return train_test_split(X, y, test_size=0.3, random_state=42)

    def apply_sampler(self, X, y, sampler):
        """Aplica o balanceamento nos dados."""
        categorical_columns = (
            [X.columns.get_loc(col) for col in X.select_dtypes(include=['object', 'category']).columns]
            if sampler.__name__ == 'balance_data_with_smotenc' else None
        )
        return sampler(X, y, categorical_features=categorical_columns) if categorical_columns else sampler(X, y)

    def train_and_evaluate(self, model, X_train, X_test, y_train, y_test) -> ModelEvaluationResult:
        """
        Trains a machine learning model and evaluates its performance using the AUC (Area Under the ROC Curve) metric.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            A machine learning model that implements `fit`, `predict`, and `predict_proba` methods.
        X_train : array-like or pandas.DataFrame
            Training dataset containing the independent variables.
        X_test : array-like or pandas.DataFrame
            Test dataset containing the independent variables.
        y_train : array-like
            True labels for the training set.
        y_test : array-like
            True labels for the test set.

        Returns
        -------
        ModelEvaluationResult
            - float: The AUC (Area Under the ROC Curve) score of the evaluated model.
            - array-like: The actual labels of the test set.
            - array-like: The model's predictions on the test set.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        return ModelEvaluationResult(auc_score, y_test, y_pred)

    def generic_pipeline(self, df, *args, **kwargs):
        """Must be implemented in the child classes."""
        raise NotImplementedError("Este método deve ser implementado na classe filha.")

    # Specific functions to be used in the child classes
    def model_raw(self, df) -> ModelEvaluationResult:
        """
        Build a model without any pr-processing

        Parameters
        ----------
        df: array-like or pandas.DataFrame
            Dataset to be processed.

        Returns
        -------
        ModelEvaluationResult
            - float: The AUC (Area Under the ROC Curve) score of the evaluated model.
            - array-like: The actual labels of the test set.
            - array-like: The model's predictions on the test set.
        """
        return self.generic_pipeline(df)

    def model_with_sampler(self, df, sampler) -> ModelEvaluationResult:
        """
        Uses pre-processing sampling technique in the data before building the model.

        Parameters
        ----------
        df: array-like or pandas.DataFrame
            Dataset to be processed.
        sampler: function in the pre_process.resampling file
            Sampling function

        Returns
        -------
        ModelEvaluationResult
            - float: The AUC (Area Under the ROC Curve) score of the evaluated model.
            - array-like: The actual labels of the test set.
            - array-like: The model's predictions on the test set.
        """
        return self.generic_pipeline(df, sampler=sampler)

    def model_with_feature_selector(self, df, feature_selector) -> ModelEvaluationResult:
        """
        Uses pre-processing feature selection technique in the data before building the model.

        Parameters
        ----------
        df: array-like or pandas.DataFrame
            Dataset to be processed.
        feature_selector: function in the pre_process.feature_selection file
            Feature Selection function

        Returns
        -------
        ModelEvaluationResult
            - float: The AUC (Area Under the ROC Curve) score of the evaluated model.
            - array-like: The actual labels of the test set.
            - array-like: The model's predictions on the test set.
        """
        return self.generic_pipeline(df, feature_selector=feature_selector)

    def model_sampler_feature_selector(self, df, sampler, feature_selector) -> ModelEvaluationResult:
        """
        Uses pre-processing sampling and feature selection technique in the data before building the model, uring the sampler first.

        Parameters
        ----------
        df: array-like or pandas.DataFrame
            Dataset to be processed.
        sampler: function in the pre_process.resampling file
            Sampling function
        feature_selector: function in the pre_process.feature_selection file
            Feature Selection function

        Returns
        -------
        ModelEvaluationResult
            - float: The AUC (Area Under the ROC Curve) score of the evaluated model.
            - array-like: The actual labels of the test set.
            - array-like: The model's predictions on the test set.
        """
        return self.generic_pipeline(df, sampler=sampler, feature_selector=feature_selector, balance_first=True)
    
    def model_feature_selector_sampler(self, df, sampler, feature_selector) -> ModelEvaluationResult:
        """
        Uses pre-processing sampling and feature selection technique in the data before building the model, using feature selection first.

        Parameters
        ----------
        df: array-like or pandas.DataFrame
            Dataset to be processed.
        sampler: function in the pre_process.resampling file
            Sampling function
        feature_selector: function in the pre_process.feature_selection file
            Feature Selection function

        Returns
        -------
        ModelEvaluationResult
            - float: The AUC (Area Under the ROC Curve) score of the evaluated model.
            - array-like: The actual labels of the test set.
            - array-like: The model's predictions on the test set.
        """
        return self.generic_pipeline(df, sampler=sampler, feature_selector=feature_selector)
