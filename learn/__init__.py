import argparse

from pipeline import Pipeline

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process pipeline commands.')
	# add arguments
	parser.add_argument('path', type=str, help='Path to .yml file with pipeline commands')
	# process args
	args = parser.parse_args()
	# run the pipeline
	pipe = Pipeline(args.path)
	pipe.run()
