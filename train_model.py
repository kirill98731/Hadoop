import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('first_exp')

def train():
    df = pd.read_csv('data/train.csv')
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as first_run:

        pipe = Pipeline([('tfidf', TfidfVectorizer()),
                         ('svc', LinearSVC())])
        pipe.fit(X_train, y_train)
        pickle.dump(pipe['tfidf'], open("tfidf.pickle", "wb"))
        preds = pipe.predict(X_test)
        clf_rep = classification_report(y_test, preds, output_dict=True)

        precision = clf_rep['positive']['precision']
        recall = clf_rep['positive']['recall']
        f1_score = clf_rep['positive']['f1-score']

        # Log any metrics you want here
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score)
        run_id = first_run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        mlflow.sklearn.log_model(pipe['svc'], artifact_path="sklearn-model", registered_model_name="sk-learn-svc-model")

if __name__ == "__main__":
    train()