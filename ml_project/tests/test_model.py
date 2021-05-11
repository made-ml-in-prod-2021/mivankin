import os
import pytest
import pandas as pd
import numpy as np
from src.models import model
from src.features import DataFaker
from utils.loggers import setup_logging, lgr, lgr_info
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
import random

@dataclass()
class test_config:
	test_eval: bool
	test_dataset_path: str
	test_dump_model_path: str
	test_load_model_path: str
	test_predict_path: str
	test_save_path: str
	test_model: str
	test_solver: str
	test_reg: str
	test_max_iter: int
	test_seed: int
	sample_size: int

def test_setup_logger():
	result = setup_logging()

	assert result[0] == 40, (f"all logger setup error {repr(result[0])}")
	assert result[1] == 20, (f"warn logger setup error {repr(result[1])}")

	lgr.handlers = []
	lgr_info.handlers = []

def test_init_model():

	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter, test_params.test_seed).model

	if test_params.test_model == 'LogisticRegression':
		assert isinstance(test_model, type(LogisticRegression())), (
			f"builded model instance is {type(test_model)}, but not sklearn LogisticRegression() instance"
		)
	elif test_params.test_model == 'GaussianNB':
		assert isinstance(test_model, type(GaussianNB())), (
			f"builded model instance is {type(test_model)}, but not sklearn GaussianNB() instance"
		)

def test_model_params():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter, test_params.test_seed).model
	params_dict = test_model.get_params()

	if test_params.test_model == 'LogisticRegression':
		assert params_dict['solver'] == test_params.test_solver, (
			f"solver must be {test_params.test_solver}, but {params_dict['solver']} found"
		)

		assert params_dict['penalty'] == test_params.test_reg, (
			f"solver must be {test_params.test_reg}, but {params_dict['penalty']} found"
		)

		assert params_dict['max_iter'] == test_params.test_max_iter, (
			f"solver must be {test_params.test_max_iter}, but {params_dict['max_iter']} found"
		)

		assert params_dict['random_state'] == test_params.test_seed, (
			f"solver must be {test_params.test_seed}, but {params_dict['random_state']} found"
		)
	elif test_params.test_model == 'GaussianNB':
		assert params_dict['var_smoothing'] == 1e-9, (
			f"Not GaussianNB instance"
		)

def test_load_dataset():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter, test_params.test_seed)
	
	test_df = test_model.load_dataset(test_params.test_dataset_path)
	
	assert isinstance(test_df, type(pd.DataFrame())), (
		f"Pandas dataframe expected, but return {type(test_df)}"
	)
	
def test_fit_without_load_data():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter, test_params.test_seed)
	try:
		error_value = test_model.split_and_fit()
	except NotImplementedError:
		error_value = -1
	
	assert -1 == error_value, (
		f"Method fit() evaluate, without dataset load"
	)

def test_fit_with_load_data():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter, test_params.test_seed)
	test_df = test_model.load_dataset(test_params.test_dataset_path)
	error_value = test_model.split_and_fit()
	
	assert 0 == error_value, (
		f"Method fit() not evaluate"
	)

def test_auc():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter, test_params.test_seed)
	test_df = test_model.load_dataset(test_params.test_dataset_path)
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

def test_predict():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter, test_params.test_seed)
	test_df = test_model.load_dataset(test_params.test_dataset_path)
	test_model.split_and_fit()

	test_sample = DataFaker(test_df)
	test_sample.ditribution_calc()
	test_sample.generate_samples(test_params.sample_size).to_csv(test_params.test_save_path, index=False)
	test_pred = test_model.predict(test_params.test_save_path, test_params.test_save_path)
	os.remove(test_params.test_save_path)
	
	assert 1 >= np.sum(list(set(test_pred))), (
		f"Expected 0 or 1 values for predictions and sum of set not greater than 1, but return {set(test_pred)} and sum {np.sum(test_pred)}"
	)

def test_all_build_model():
	test_schema = class_schema(test_config)
	test_inst = test_schema()

	with open('test_cfg.yaml', 'r') as f:
		test_params = test_inst.load(yaml.safe_load(f))

	test_sample = DataFaker(pd.read_csv(test_params.test_dataset_path))
	test_sample.ditribution_calc()
	test_sample = test_sample.generate_samples(test_params.sample_size)
	test_sample.insert(test_sample.shape[1], 'target', [int(random.randint(0, 1)) for i in range(test_params.sample_size)])
	test_sample.to_csv('test_fake.csv')

	test_model = model.UCImodel(test_params.test_model, test_params.test_solver, test_params.test_reg, test_params.test_max_iter,
								test_params.test_seed)


	a = test_model.load_dataset('test_fake.csv')

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

