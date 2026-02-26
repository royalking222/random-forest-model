import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score


path = "/kaggle/input/creditcardfraud/creditcard.csv"
file = pd.read_csv(path)

# feature and target set
x = file.drop(["Class"], axis=1)
y = file["Class"]

# train

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size = 0.02,
    random_state = 42
)

# sclaing
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# model preparing
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight = "balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
y_prob = rf_model.predict_proba(x_test)[:,1]

# compare actual value and predict value
compare = pd.DataFrame({
    "predict" : y_pred[:20],
    "actual" : y_test[:20]
})
print(compare)

# evaluation
print("precision:", precision_score(y_test , y_pred) * 100,"%" )
print("Recall   :", recall_score(y_test, y_pred)* 100,"%")
print("F1 Score :", f1_score(y_test, y_pred)* 100,"%")
print("ROC AUC  :", roc_auc_score(y_test, y_prob)* 100,"%")

print("no errors")