import os
import urllib.request
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,auc
from sklearn.preprocessing import StandardScaler

SOURCE_URL = 'https://www.openml.org/data/get_csv/1595261/phpMawTba.csv'
FILENAME = 'salary_prediction.csv'
DATA_DIR= 'data'

MAX_FEATURES = 10
STEP_DEL_FEATURES = 2
RDM = 1
TEST_SIZE = 0.2

def create_dir(data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

def download_data(source: str, dest: str):
    print("Downloading data...")
    urllib.request.urlretrieve(source, dest)

def _data_preparation(dest: str):
    df = pd.read_csv(dest)
    df["class"] = df["class"].apply(lambda x: 0 if x.strip() == '<=50K' else 1)
    numerical = [elem for elem in df._get_numeric_data().columns if elem not in ["class"]]
    categorical = [feature for feature in list(df.columns) if feature not in numerical+["class"]]

    # One hot encoding: Convert categorical to numerical
    df_ml = df.copy()
    encoded = pd.get_dummies(df_ml.loc[:,categorical])
    df_ml = pd.concat([df_ml.loc[:,numerical],encoded],axis = 1)
    target = df.loc[:,"class"]

    print(f"Shape of the original dataset: {df.shape}\nShape of one-hot encoded dataset: {df_ml.shape}")

    X_train, X_test, y_train, y_test = train_test_split(df_ml, target, test_size=TEST_SIZE, random_state=RDM)

    # Feature selection: Select which variables are best to predict "class"
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None)
    selector = RFE(model, MAX_FEATURES, step=STEP_DEL_FEATURES, verbose=0)
    selector = selector.fit(X_train.values, y_train.values)

    selected_features = X_train.loc[:,selector.support_].columns.values.tolist()
    print("\nThe {} variables that predict the best are: \n{}".format(MAX_FEATURES, ',\n'.join(selected_features)))

    X_train.loc[:,selector.support_].to_csv(DATA_DIR+'/processed/'+'X_train.csv', index=False)
    X_test.loc[:,selector.support_].to_csv(DATA_DIR+'/processed/'+'X_test.csv', index=False)
    
    y_train.to_csv(DATA_DIR+'/processed/'+'y_train.csv', index=False, header=True)
    y_test.to_csv(DATA_DIR+'/processed/'+'y_test.csv', index=False, header=True)

    return X_train, X_test, y_train, y_test


def _data_normalisation():

    X_train = pd.read_csv(DATA_DIR+'/processed/'+"X_train.csv")
    X_test = pd.read_csv(DATA_DIR+'/processed/'+"X_test.csv")

    y_train = pd.read_csv(DATA_DIR+'/processed/'+"y_train.csv")
    y_test = pd.read_csv(DATA_DIR+'/processed/'+"y_test.csv")

    print("\n\nTraining a RandomForest classifier:\nn_estimators=100,\ncriterion=gini\nmax_features=4\n\n")
    model = RandomForestClassifier(n_estimators=200, criterion='gini', max_features=4, max_depth=None)
    model.fit(X_train.values,y_train.values.flatten())

    y_predicted = model.predict(X_test.values)

    tn, fp, fn, tp = confusion_matrix(y_test,y_predicted).ravel()
    print("Confusion Matrix:\n")
    print("tp: {} values are classified positive and are positive".format(tp))
    print("tn: {} values are classified negative and are negative".format(tn))
    print("fp: {} values predicted as positive but actually were negative".format(fp))
    print("fn: {} values predicted as negative but actually were positive".format(fn))

    print()
    print("Accuracy: {}".format(accuracy_score(y_test,y_predicted)))
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    print("AUC: {}".format(auc(fpr,tpr)))

    # Scaling
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RDM)

    X_train.to_csv(DATA_DIR+'/processed/'+'X_train_norm.csv', index=False, header=False)
    X_test.to_csv(DATA_DIR+'/processed/'+'X_test_norm.csv', index=False, header=False)
    
    y_train.to_csv(DATA_DIR+'/processed/'+'y_train_norm.csv', index=False, header=False)
    y_test.to_csv(DATA_DIR+'/processed/'+'y_test_norm.csv', index=False, header=False)


if __name__ == "__main__":
    create_dir(DATA_DIR)
    create_dir(DATA_DIR+'/raw')
    create_dir(DATA_DIR+'/processed')

    download_data(SOURCE_URL, DATA_DIR+'/raw/'+FILENAME)
    _data_preparation(DATA_DIR+'/raw/'+FILENAME)
    _data_normalisation()

    #X_train, X_test, y_train, y_test = data_preparation( DATA_DIR+'/raw/'+FILENAME)