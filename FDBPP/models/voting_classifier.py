from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from models.base import GenericModelBase

class VotingEnsembleModel(GenericModelBase):
    def generic_pipeline(self, df, sampler=None, feature_selector=None, threshold=0.5, k=10, balance_first=False, voting='soft'):
        X_train, X_test, y_train, y_test = self.prepare_data(df, sampler, feature_selector, threshold, k, balance_first)

        estimators = [
            ('cart', DecisionTreeClassifier(random_state=42)),
            ('knn', KNeighborsClassifier()),
            ('lr', LogisticRegression(max_iter=500, random_state=42)),
            ('nb', GaussianNB()),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('svm', SVC(probability=True, random_state=42))
        ]

        model = VotingClassifier(estimators=estimators, voting=voting)
        return self.train_and_evaluate(model, X_train, X_test, y_train, y_test)
