import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

masked_sensors_gz = [53, 4, 8, 47, 43, 44, 17, 34, 42, 38, 7, 21, 54]
masked_sensors_bj = [32, 9, 4, 3, 19, 24, 18]

config = {
    'logs': 'logs/'
}
datasets_path = {
    'air': 'datasets/air_quality',
    'la': 'datasets/metr_la',
    'bay': 'datasets/pems_bay',
    'synthetic': 'datasets/synthetic',
    'air_city': 'datasets/air_city'
}
epsilon = 1e-8

for k, v in config.items():
    config[k] = os.path.join(base_dir, v)
for k, v in datasets_path.items():
    datasets_path[k] = os.path.join(base_dir, v)
