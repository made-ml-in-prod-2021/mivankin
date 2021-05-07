import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


from sklearn.pipeline import FeatureUnion, Pipeline
from features import FeaturesTransformer

from utils import loggers as lg

DEFAULT_DATASET_PATH = 'data/heart.csv'
DEFAULT_MODEL_PATH = 'model.pkl'

INFERENCE_DATASET_PATH = 'test.csv'

OUT_PREDICTIONS_PATH = 'preds.out'


class UCImodel:
    def __init__(self, solver='liblinear', penalty='l2', max_iter=100, random_state=42):
        lg.lgr_info.info("Init model...")

        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X = None
        self.y = None
        
        self.model = LogisticRegression(
                                solver=solver, 
                                penalty=penalty, 
                                max_iter=max_iter, 
                                random_state=random_state
                                )

    def load_dataset(self, path: str):
        lg.lgr_info.info("Loading dataset...")
        try:
            self.dataset = pd.read_csv(path)
        except pd.errors.ParserError:
            lg.lgr.error("ParserError, check dataset type, csv dataset expected")
        except FileNotFoundError:
            lg.lgr.error("FileNotFoundError, dataset not found, check it")

        try:
            self.X = pd.DataFrame(np.c_[
            self.dataset.age,
            self.dataset.sex,
            self.dataset.cp, 
            self.dataset.restecg, 
            self.dataset.thalach, 
            self.dataset.exang,
            self.dataset.chol,
            self.dataset.slope,
            self.dataset.ca,
            self.dataset.thal
            ])

            self.y = self.dataset.target
        except AttributeError:
            lg.lgr.error(f"Invalid data for Heart Disease UCI classification, check columns headers, or data from {path}")
            return None
            
        return self.dataset
    
    def split_and_fit(self, test_size=0.2, random_state=42):
        lg.lgr_info.info("Splitting data...")
        _exec_value = 0
        try:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                                                                        self.X, self.y,
                                                                        test_size=test_size,
                                                                        random_state=random_state
                                                                    )
        except TypeError:
            lg.lgr.error(f"Expected sequence or array-like, got for X {type(self.X)} and for y {type(self.y)}")
            _exec_value = -1
            return _exec_value
        except ValueError:
            lg.lgr.error(f"The resulting train set will be empty. Adjust any of the aforementioned parameters or check data size.")
            _exec_value = -1
            return _exec_value

        self.__fit()

        return _exec_value

    def __fit(self):
        lg.lgr_info.info("Fitting model...")
        self.model.fit(self.X_train, self.Y_train)
        lg.lgr_info.info("Checking fit...")
        train_auc, test_auc = self.auc()
        lg.lgr_info.info(f"AUC score: train is {np.round(train_auc, 4)}, test is {np.round(test_auc, 4)}")
    
    def auc(self):
        lg.lgr_info.info("AUC calculating...")
        auc_values = list()
        for name, X, y, model in [
            ('train', self.X_train, self.Y_train, self.model),
            ('test ', self.X_test, self.Y_test, self.model)
        ]:
            proba = model.predict_proba(X)[:, 1]
            auc_values.append(roc_auc_score(y, proba))
            
        return auc_values

    def predict(self, queries, path):
        lg.lgr_info.info("reading csv for predictions...")
        queries = pd.read_csv(queries)
        lg.lgr_info.info("Wait for predictions...")

        preds = self.model.predict(queries)

        lg.lgr_info.info(f"Dump predictions to {path}...")
        np.savetxt(path, preds, fmt="%d")
        return preds

def dump_model(obj, path):
    lg.lgr_info.info("Dump model...")
    with open(path, "wb") as file:
        pickle.dump(obj, file)

def load_model(path):
    lg.lgr_info.info("Loading model...")
    with open(path, "rb") as file:
        return pickle.load(file)

def callback_build(arguments):
    uci_model = UCImodel()
    uci_model.load_dataset(arguments.dataset_path)

    pipeline = Pipeline(steps=[('features_pipeline', FeaturesTransformer(uci_model.X.columns))])
    pipeline.fit(uci_model.X)

    uci_model.split_and_fit()
    dump_model(uci_model, arguments.dump_model_path)
    lg.lgr_info.info("UCI model builded")

def callback_predict(arguments):
    uci_model = load_model(arguments.load_model_path)
    uci_model.predict(arguments.predict_path, arguments.save_path)
    lg.lgr_info.info(f"Predictions calculated and dumped to {arguments.save_path}, check it")