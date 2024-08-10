import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import optuna
import joblib
import xgboost as xgb

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 270)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

main_df = pd.read_csv('Datasets/Hotel_Reservations.csv')
df = main_df.copy()

df.head()

def extract_columns(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['object', 'category']]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes not in ['object', 'category']]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes in ['object', 'category']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes not in ['object', 'category']]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print("Observations:", dataframe.shape[0])
    print("Variables:", len(dataframe.columns))
    print("cat_cols:", len(cat_cols))
    print("num_cols:", len(num_cols))
    print("cat_but_car:", len(cat_but_car))
    print("num_but_cat:", len(num_but_cat))

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.ylabel("Count")
        plt.title(numerical_col)
        plt.show(block=True)


def target_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


#################
# Data Preprocessing
#################
cat_cols, num_cols, cat_but_car = extract_columns(df)

to_num_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_previous_cancellations', 'no_of_special_requests']
cat_cols = [col for col in cat_cols if col not in to_num_cols]

to_cat_cols = ['arrival_month']
num_cols = [col for col in num_cols if col not in to_cat_cols]

for x in to_cat_cols:
    cat_cols.append(x)

for x in to_num_cols:
    num_cols.append(x)

target = ['booking_status']

#################
# Extraction
#################

new_df = df.copy()

# Total Duration
new_df['visit_duration'] = new_df['no_of_week_nights'] + new_df['no_of_weekend_nights']

#########################
# Model Testing
#########################

#################
# Base Model
#################
df_encoded = new_df.copy()
df_encoded['booking_status'] = df_encoded['booking_status'].apply(lambda x: 1 if x == "Canceled" else 0)
cat_to_label = [col for col in df_encoded.columns if df_encoded[col].dtypes in ['object', 'category']]
cat_to_label = [col for col in cat_to_label if col not in "Booking_ID"]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in cat_to_label:
    label_encoder(df_encoded, col)

df_encoded.drop(['Booking_ID', 'no_of_weekend_nights', 'no_of_week_nights', 'arrival_year',
                 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled'], axis=1, inplace=True)


scalers = {}
for col in df_encoded.columns:
    if col not in "booking_status" and col not in cat_to_label:
        scaler = StandardScaler()
        df_encoded[col] = scaler.fit_transform(df_encoded[[col]])
        scalers[col] = scaler


X = df_encoded.drop(['booking_status'], axis=1)
y = df_encoded['booking_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## Oversampling
adasyn = ADASYN(sampling_strategy='minority', n_neighbors=5)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
pd.Series(y_res).value_counts()
##

###################
# Hyperparameter Optimization with Optuna
###################

# RandomForest
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    max_features = trial.suggest_int('max_features', 2, 20)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)

    return cross_val_score(clf, X_res, y_res, cv=3, n_jobs=-1).mean()


study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=300)

print("Best parameters: ", study_rf.best_params)
print("Best cross-validation accuracy: ", study_rf.best_value)


# XGBoost
def objective_xgb(trial):
    param = {
        'verbosity': 0,
        'objective': 'multi:softmax',
        'num_class': 3,
        'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.7, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.7, 1.0]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    clf = XGBClassifier(**param)

    return cross_val_score(clf, X_res, y_res, cv=3, n_jobs=-1).mean()


study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=300)

print("Best parameters: ", study_xgb.best_params)
print("Best cross-validation accuracy: ", study_xgb.best_value)


# For RandomForest
best_rf = RandomForestClassifier(**study_rf.best_params)
best_rf.fit(X_res, y_res)
print("Test accuracy: ", best_rf.score(X_test, y_test))
y_pred = best_rf.predict(X_test)
print(classification_report(y_test, y_pred))

# For XGBoost
best_xgb = XGBClassifier(**study_xgb.best_params)
best_xgb.fit(X_res, y_res)
print("Test accuracy: ", best_xgb.score(X_test, y_test))
y_pred = best_xgb.predict(X_test)
print(classification_report(y_test, y_pred))


# Cross Validation
rf_cv = cross_validate(best_rf, X_test, y_test, cv=5, n_jobs=-1, scoring=['accuracy', 'roc_auc', 'f1'])
rf_cv['test_accuracy'].mean()
rf_cv['test_roc_auc'].mean()
rf_cv['test_f1'].mean()

xgb_cv = cross_validate(best_xgb, X_test, y_test, cv=5, n_jobs=-1, scoring=['accuracy', 'roc_auc', 'f1'])
xgb_cv['test_accuracy'].mean()
xgb_cv['test_roc_auc'].mean()
xgb_cv['test_f1'].mean()


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()


plot_confusion_matrix(y_test, y_pred)

#############
# Model File
#############
joblib.dump(best_rf, 'rf.pkl')
joblib.dump(best_xgb, 'xgb.pkl')
joblib.dump(scalers, 'scalers.pkl')

############
# Tree Visualization
############
model = joblib.load('xgb.pkl')

booster = model.get_booster()

graph_data = xgb.to_graphviz(booster, num_trees=0, rankdir='TB',
                             condition_node_params={'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e48068'},
                             leaf_node_params={'shape': 'box', 'style': 'filled', 'fillcolor': '#8d85e8'})

graph_data.render("xgb_tree")
