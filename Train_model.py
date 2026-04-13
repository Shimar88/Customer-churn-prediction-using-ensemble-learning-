from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib

from preprocessing import load_and_preprocess

X, y = load_and_preprocess("../data/Telco-Customer-Churn.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier()
gb = GradientBoostingClassifier()

model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "../models/churn_model.pkl")
