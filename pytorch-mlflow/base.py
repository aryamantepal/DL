import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load iris dataset
X, y = datasets.load_iris(return_X_y=True)

# train, test data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


## log model & metadata to MLflow

# set up tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# create a new mlflow experiment
mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():
    # log the hyperparameters
    mlflow.log_params(params)
    
    # log the loss metric
    mlflow.log_metric("accuracy", accuracy)
    
    # infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))
    
    # log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature = signature,
        input_example=X_train,
        registered_model_name="tutorial for basic model"
    )
    
    # set a tag to remind us what we used the model for
    
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )
    
    
# we can perform inference by using MLflow's pyfunc flavor to load the model. Then, we run predict on new data using the loaded model
load_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = load_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result['actual_class'] = y_test
result['predicted_class'] = predictions

print(result[:4])