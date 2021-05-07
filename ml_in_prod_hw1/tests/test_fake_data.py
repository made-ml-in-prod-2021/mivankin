import pytest
from src.features import fake_data
import pandas as pd
import random


def test_fake_data_instance():
	test_df = pd.DataFrame()
	test_fake = fake_data(test_df)
	
	assert isinstance(test_fake.data, type(pd.DataFrame())), (
		f"Expected pandas dataframe in fake_data, but return {type(test_fake.data)}"
	)
	
	assert test_fake.data.shape == test_df.shape, (
		f"Expected fake_data shape {test_fake.data.shape} != data shape for build {test_df.shape}"
	)	

def test_ditribution_calc():
	test_df = pd.DataFrame()
	test_fake = fake_data(test_df)
	
	assert test_fake.distr_params.shape == (0, 2), (
		f"Expected pandas dataframe in fake_data, but return {type(test_fake.data)}"
	)	
	
def test_generate_samples():
	sample_size = 100
	test_df = pd.DataFrame([random.randint(0, i) for i in range(sample_size)])
	
	test_sample = fake_data(test_df)
	test_sample.ditribution_calc()
	test_sample = test_sample.generate_samples(sample_size)
	
	assert test_df.shape == test_sample.shape, (
		f"Faked data shape != real data shape, expected {test_df.shape}, but generate {test_sample.shape}"
	)
	

def test_check_ks_2samp():
	sample_size = 100
	test_df = pd.DataFrame([random.randint(0, i) for i in range(sample_size)])
	another_test_df = pd.DataFrame([random.randint(0, i) for i in range(sample_size, sample_size * 2)])
	test_sample = fake_data(test_df)
	test_sample.ditribution_calc()
	test_sample.generate_samples(sample_size)
	
	assert test_sample.check_ks_2samp(), (
		f"Original distribution is not equal faked distribution"
	)