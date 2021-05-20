from scipy.stats import multivariate_normal, ks_2samp
from tests.features import DataFaker

SAMPLE_SIZE = 1000

def validate(model, data):
    test_sample = DataFaker(model.dataset)
    test_sample.ditribution_calc()
    test_data = test_sample.generate_samples(SAMPLE_SIZE)

    for col in test_sample.samples_dict.keys():
        d_p_values = ks_2samp(data[col], test_sample.samples_dict[col])
        if d_p_values[0] == 1:
            print("Generated distributuion for tests not equal original distributuion")
            return False, col

    return True, ''