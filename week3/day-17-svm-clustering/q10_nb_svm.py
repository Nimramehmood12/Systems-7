from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
a, b, c, d = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
for name, model in [("Naive Bayes", GaussianNB()), ("SVM", make_pipeline(StandardScaler(), SVC()))]:
    pred = model.fit(a, c).predict(b)
    print(name, "precision", precision_score(d, pred), "recall", recall_score(d, pred))
