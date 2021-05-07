from scipy.stats import multivariate_normal, ks_2samp
import random
import pandas as pd
import numpy as np

CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
SCIPED = ['fbs', 'oldpeak', 'trestbps', 'target']

class fake_data:
	def __init__(self, input_df, categorical_cols=CATEGORICAL, scip_cols=SCIPED):
		self.data = input_df
		self.columns = list(input_df.columns)
		self.samples_dict = dict()
		self.categorical_cols = categorical_cols
		self.scip_cols = set(scip_cols)
		self.distr_params = np.zeros((self.data.shape[1], 2))
		
	def ditribution_calc(self):
		for i, col in enumerate(self.columns):
			if col not in self.scip_cols:
				self.distr_params[i][0] = np.mean(self.data[col]) 
				self.distr_params[i][1] = np.cov(self.data[col]) 
		return 
		
	def generate_samples(self, sample_size=1):
		pd_cols = list()
		for i, col in enumerate(self.columns):
			if col not in self.scip_cols:
				pd_cols.append(col)
				distr = multivariate_normal(mean=self.distr_params[i][0], cov=self.distr_params[i][1], allow_singular=False)
				if col not in set(self.categorical_cols):
					sampling = distr.rvs(sample_size)
				else:
					sampling = np.abs((distr.rvs(sample_size)).astype(int))
				
				if sample_size == 1:
					self.samples_dict[col] = np.array([sampling])
				else:
					self.samples_dict[col] = sampling
					
		return pd.DataFrame(self.samples_dict, columns=pd_cols)
		
	def check_ks_2samp(self):
		for col in self.samples_dict.keys():
			d_p_values= ks_2samp(self.data[col], self.samples_dict[col])
			if d_p_values[0] == 1:
				lg.lgr.exception("Generated distributuion for tests not equal original distributuion")
				return False
			
		return True
