import os
import logging
import pytest
import pandas as pd
import numpy as np
from src.models import model, callback_build, callback_predict
from tests.features import DataFaker
from utils.loggers import setup_logging, lgr, lgr_info
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
import random

LOGGER = logging.getLogger(__name__)

@dataclass()
class test_config:
	eval: bool
	dataset_path: str
	dump_model_path: str
	load_model_path: str
	predict_path: str
	save_path: str
	model: str
	solver: str
	reg: str
	max_iter: int
	seed: int
	sample_size: int

@pytest.fixture(scope='module')
def init_model():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	return model.UCImodel(test_params.model, test_params.solver, test_params.reg, test_params.max_iter, test_params.seed), test_params



def test_setup_logger():
	result = setup_logging()

	assert result[0] == 40, (f"all logger setup error {repr(result[0])}")
	assert result[1] == 20, (f"warn logger setup error {repr(result[1])}")

	lgr.handlers = []
	lgr_info.handlers = []

def test_init_model(init_model):
	test_model, test_params = init_model

	if test_params.model == 'LogisticRegression':
		assert isinstance(test_model.model, type(LogisticRegression())), (
			f"builded model instance is {type(test_model.model)}, but not sklearn LogisticRegression() instance"
		)
	elif test_params.model == 'GaussianNB':
		assert isinstance(test_model.model, type(GaussianNB())), (
			f"builded model instance is {type(test_model.model)}, but not sklearn GaussianNB() instance"
		)

def test_model_params(init_model):
	test_model, test_params = init_model

	params_dict = test_model.model.get_params()

	if test_params.model == 'LogisticRegression':
		assert params_dict['solver'] == test_params.solver, (
			f"solver must be {test_params.solver}, but {params_dict['solver']} found"
		)

		assert params_dict['penalty'] == test_params.reg, (
			f"solver must be {test_params.reg}, but {params_dict['penalty']} found"
		)

		assert params_dict['max_iter'] == test_params.max_iter, (
			f"solver must be {test_params.max_iter}, but {params_dict['max_iter']} found"
		)

		assert params_dict['random_state'] == test_params.seed, (
			f"solver must be {test_params.seed}, but {params_dict['random_state']} found"
		)
	elif test_params.model == 'GaussianNB':
		assert params_dict['var_smoothing'] == 1e-9, (
			f"Not GaussianNB instance"
		)

def test_load_dataset(init_model):
	test_model, test_params = init_model
	test_df = test_model.load_dataset(test_params.dataset_path)
	
	assert isinstance(test_df, type(pd.DataFrame())), (
		f"Pandas dataframe expected, but return {type(test_df)}"
	)
	
def test_fit_without_load_data(init_model):
	test_model, test_params = init_model
	test_model.X = []
	test_model.y = []

	try:
		error_value = test_model.split_and_fit()
	except NotImplementedError:
		error_value = -1
	
	assert -1 == error_value, (
		f"Method fit() evaluate, without dataset load"
	)

def test_fit_with_load_data(init_model):
	test_model, test_params = init_model

	test_df = test_model.load_dataset(test_params.dataset_path)
	error_value = test_model.split_and_fit()
	
	assert 0 == error_value, (
		f"Method fit() not evaluate"
	)

def test_auc(init_model):
	test_model, test_params = init_model

	test_df = test_model.load_dataset(test_params.dataset_path)
	test_model.split_and_fit()
	
	train_auc, test_auc = test_model.auc()
	
	assert 0 < train_auc, (
		f"train AUC value {train_auc} is negative"
	)

	assert 0 < test_auc, (
		f"test AUC value {test_auc} is negative"
	)
	
	assert 1 > train_auc, (
		f"train AUC value {train_auc} greater than 1"
	)

	assert 1 > test_auc, (
		f"test AUC value {test_auc} greater than 1"
	)

def test_predict(init_model):
	test_model, test_params = init_model

	test_df = test_model.load_dataset(test_params.dataset_path)
	test_model.split_and_fit()

	test_sample = DataFaker(test_df)
	test_sample.ditribution_calc()
	test_sample.generate_samples(test_params.sample_size).to_csv(test_params.save_path, index=False)
	test_pred = test_model.predict(test_params.save_path, test_params.save_path)
	os.remove(test_params.save_path)
	
	assert 1 >= np.sum(list(set(test_pred))), (
		f"Expected 0 or 1 values for predictions and sum of set not greater than 1, but return {set(test_pred)} and sum {np.sum(test_pred)}"
	)

def test_all_build_model(caplog):
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	with caplog.at_level(logging.INFO):
		callback_build(test_params)

	assert 'UCI model builded' in caplog.text

def test_all_predict_model(caplog):
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	with caplog.at_level(logging.INFO):
		callback_predict(test_params)

	assert f"Predictions calculated and dumped to {test_params.save_path}, check it" in caplog.text

