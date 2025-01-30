from sklearn.ensemble import RandomForestClassifier
from models.base import GenericModelBase

class RandomForestModel(GenericModelBase):
    def generic_pipeline(self, df, sampler=None, feature_selector=None, threshold=0.5, k=10, balance_first=False):
        X_train, X_test, y_train, y_test = self.prepare_data(df, sampler, feature_selector, threshold, k, balance_first)
        model = RandomForestClassifier(random_state=42)
        return self.train_and_evaluate(model, X_train, X_test, y_train, y_test)
