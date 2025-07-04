import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
ab = AdaBoostClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(eval_metric='logloss')

rf.fit(X_train, y_train)
ab.fit(X_train, y_train)
xgb.fit(X_train, y_train)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('ab', ab), ('xgb', xgb)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

params = {
    'rf__n_estimators': [100, 200],
    'ab__n_estimators': [50, 100],
}

grid = GridSearchCV(ensemble, param_grid=params, cv=3)
grid.fit(X_train, y_train)


joblib.dump(ensemble, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

