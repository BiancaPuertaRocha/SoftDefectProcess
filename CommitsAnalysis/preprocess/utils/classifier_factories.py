
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# -------- Classifier Factories --------

def create_rf(random_state=42): return RandomForestClassifier(n_estimators=50, random_state=random_state)

def create_bagging(base, random_state=42): return BaggingClassifier(estimator=base, n_estimators=10, random_state=random_state)

def create_cart(random_state=42): return DecisionTreeClassifier(random_state=random_state)

def create_voting(random_state=42):
    return VotingClassifier(estimators=[
        ('cart', create_cart(random_state)),
        ('knn', KNeighborsClassifier()),
        ('lr', LogisticRegression(max_iter=1000, random_state=random_state)),
        ('nb', GaussianNB()),
        ('rf', create_rf(random_state)),
        ('svm', SVC(probability=True, random_state=random_state))
    ], voting='soft')