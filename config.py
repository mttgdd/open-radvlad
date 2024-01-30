import yaml
import argparse

def get_cfg_impl(config_file):
	with open(config_file) as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)
	return cfg

def get_cfg():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)

	args = parser.parse_args()

	return get_cfg_impl(args.config)