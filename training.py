import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib  # To save the models
from river import ensemble
from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier

# Load data
print("Loading data...")
df_ids2018 = pd.read_csv('cleaned_ids2018_sampled.csv')

# Check if labels are numeric or strings and map if necessary
if df_ids2018['Label'].dtype == 'object':
    label_dict = {
        'Benign': 1,
        'FTP-BruteForce': 2,
        'SSH-Bruteforce': 3,
        'DDOS attack-HOIC': 4,
        'Bot': 5,
        'DoS attacks-GoldenEye': 6,
        'DoS attacks-Slowloris': 7,
        'DDOS attack-LOIC-UDP': 8,
        'Brute Force -Web': 9,
        'Brute Force -XSS': 10,
        'SQL Injection': 11
    }
    print("Mapping string labels to numeric labels...")
    df_ids2018['Label'] = df_ids2018['Label'].map(label_dict)

# Log original label distribution
print("Original Label distribution:\n", df_ids2018['Label'].value_counts())

# Use the 12 selected features
selected_features = [
    'Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Bwd Pkt Len Max', 'Flow Pkts/s',
    'Fwd IAT Mean', 'Bwd IAT Tot', 'Bwd IAT Mean', 'RST Flag Cnt',
    'URG Flag Cnt', 'Init Fwd Win Byts', 'Fwd Seg Size Min', 'Idle Max'
]

X = df_ids2018[selected_features]
y = df_ids2018['Label']

# Split data into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Log label distribution in training set
print("Label distribution in training set before resampling:\n", Counter(y_train))

# Apply undersampling to reduce major attack classes
benign_count = int(len(y_train) * 0.5)
attack_count = len(y_train) - benign_count

under_strategy = {
    1: benign_count,
    2: min(int(attack_count / 10), Counter(y_train)[2]),
    3: min(int(attack_count / 10), Counter(y_train)[3]),
    4: min(int(attack_count / 10), Counter(y_train)[4]),
    5: min(int(attack_count / 10), Counter(y_train)[5]),
    6: min(int(attack_count / 10), Counter(y_train)[6]),
    7: min(int(attack_count / 10), Counter(y_train)[7]),
    8: min(int(attack_count / 10), Counter(y_train)[8]),
    9: min(int(attack_count / 10), Counter(y_train)[9]),
    10: min(int(attack_count / 10), Counter(y_train)[10]),
    11: min(int(attack_count / 10), Counter(y_train)[11])
}

under_sampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

# Log label distribution after undersampling
print("Label distribution after undersampling:\n", Counter(y_train_under))

# Apply SMOTE for oversampling minority attack classes
smote_strategy = {
    1: benign_count,
    2: min(int(attack_count / 10), Counter(y_train_under)[2]),
    3: min(int(attack_count / 10), Counter(y_train_under)[3]),
    4: min(int(attack_count / 10), Counter(y_train_under)[4]),
    5: min(int(attack_count / 10), Counter(y_train_under)[5]),
    6: min(int(attack_count / 10), Counter(y_train_under)[6]),
    7: min(int(attack_count / 10), Counter(y_train_under)[7]),
    8: min(int(attack_count / 10), Counter(y_train_under)[8]),
    9: min(int(attack_count / 10), Counter(y_train_under)[9]),
    10: min(int(attack_count / 10), Counter(y_train_under)[10]),
    11: min(int(attack_count / 10), Counter(y_train_under)[11])
}

smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_under, y_train_under)

# Subtract 1 from the labels for XGBoost
y_train_res_xgb = y_train_res - 1
y_test_xgb = y_test - 1

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

# Save the Random Forest model
joblib.dump(rf_model, 'rf_model.joblib')
print("Random Forest model saved as 'rf_model.joblib'")

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest classification report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42, use_label_encoder=False)
xgb_model.fit(X_train_res, y_train_res_xgb)

# Save the XGBoost model
joblib.dump(xgb_model, 'xgb_model.joblib')
print("XGBoost model saved as 'xgb_model.joblib'")

# Predict and evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
print("XGB Accuracy:", accuracy_score(y_test_xgb, y_pred_xgb))
print("XGBoost classification report:\n", classification_report(y_test_xgb, y_pred_xgb))
print("XGBoost confusion matrix:\n", confusion_matrix(y_test_xgb, y_pred_xgb))

# Convert to dictionary format for River (online learning library)
X_train_dict = X_train_res.to_dict(orient='records')
X_test_dict = X_test.to_dict(orient='records')

# Train and evaluate Online Random Forest (ORF) with streaming data
print("Training Online Random Forest (ORF) model incrementally...")
orf_model = HoeffdingAdaptiveTreeClassifier()

for xi, yi in zip(X_train_dict, y_train_res):
    orf_model.learn_one(xi, yi)

# Stream data to ORF and predict
print("Evaluating ORF model with streaming data...")
orf_preds = []
for xi, yi in zip(X_test_dict, y_test):  # Stream test data one by one
    orf_pred = orf_model.predict_one(xi)
    orf_preds.append(orf_pred)
    orf_model.learn_one(xi, yi)  # Update the model incrementally after each prediction

print("ORF Accuracy:", accuracy_score(y_test, orf_preds))
print("ORF Confusion Matrix:\n", confusion_matrix(y_test, orf_preds))

# Save the ORF model
joblib.dump(orf_model, 'orf_model.joblib')
print("ORF model saved as 'orf_model.joblib'")

# Train and evaluate Hoeffding Tree (HT) with streaming data
print("Training Hoeffding Tree (HT) model incrementally...")
ht_model = HoeffdingTreeClassifier()

for xi, yi in zip(X_train_dict, y_train_res):
    ht_model.learn_one(xi, yi)

# Stream data to HT and predict
print("Evaluating HT model with streaming data...")
ht_preds = []
for xi, yi in zip(X_test_dict, y_test):  # Stream test data one by one
    ht_pred = ht_model.predict_one(xi)
    ht_preds.append(ht_pred)
    ht_model.learn_one(xi, yi)  # Update the model incrementally after each prediction

print("HT Accuracy:", accuracy_score(y_test, ht_preds))
print("HT Confusion Matrix:\n", confusion_matrix(y_test, ht_preds))

# Save the HT model
joblib.dump(ht_model, 'ht_model.joblib')
print("HT model saved as 'ht_model.joblib'")

# Custom evaluation: Pass one record per label to all four models and print predictions
print("Performing custom evaluation for all models...")
unique_labels = y_test.unique()
for label in unique_labels:
    sample_record = X_test[y_test == label].iloc[0].to_dict()
    print(f"Record for label {label}:\n", sample_record)

    # Prediction using Random Forest
    rf_pred = rf_model.predict([list(sample_record.values())])
    print(f"Random Forest prediction for label {label}: {rf_pred}")

    # Prediction using XGBoost
    xgb_pred = xgb_model.predict([list(sample_record.values())])
    print(f"XGBoost prediction for label {label}: {xgb_pred}")

    # Prediction using ORF
    orf_pred = orf_model.predict_one(sample_record)
    print(f"ORF prediction for label {label}: {orf_pred}")

    # Prediction using HT
    ht_pred = ht_model.predict_one(sample_record)
    print(f"HT prediction for label {label}: {ht_pred}")

# End of process
print("Process complete.")


