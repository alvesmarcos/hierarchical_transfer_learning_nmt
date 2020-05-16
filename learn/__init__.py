import argparse
import logging
import os
import sys
import time

from pipeline import Pipeline

# create logger
logger = logging.getLogger('pipeline')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.FileHandler(
	os.path.join('logs', f'pipeline {time.strftime("%Y-%m-%d@%H.%M.%S")}.log')
)
ch.setLevel(logging.DEBUG)
# Configure log output.
logging.basicConfig(
    format='[%(levelname)s] [%(asctime)s] %(message)s',
    level=logging.INFO,
	handlers=[logging.StreamHandler(sys.stdout), ch]
)

if __name__ == '__main__':
	logger.info('Hierarchical Transfer Learning Pipeline')
	parser = argparse.ArgumentParser(description='Process pipeline commands.')
	# add arguments
	parser.add_argument('path', type=str, help='Path to .yml file with pipeline commands')
	# process args
	args = parser.parse_args()
	# run the pipeline
	logger.info(f'Command file received {args.path}')
	pipe = Pipeline(args.path)
	pipe.run()
