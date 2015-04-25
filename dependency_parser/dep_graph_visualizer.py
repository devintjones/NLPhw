import random
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
from providedcode.dependencycorpusreader import DependencyCorpusReader

import sys
import os

if __name__ == '__main__':
    
	if sys.argv[1] not in ['english','danish','korean','swedish']:
		sys.exit('Usage: must call either englihs, danish, korean, or swedish')
	lang   = sys.argv[1]
	base_path = '/home/coms4705/Documents/Homework2/data/'
	lang_base = {'english'      :'english',
			'korean' :'korean',
			'danish' :'danish/ddt',
			'swedish':'swedish/talbanken05'}

	file_name = {'english' : 'train/en-universal-train.conll',
			'korean' : 'train/ko-universal-train.conll',
			'danish' : 'train/danish_ddt_train.conll',
			'swedish' : 'train/swedish_talbanken05_train.conll'}

	file_path = os.path.join(base_path,lang_base.get(lang),file_name.get(lang))

	os.system('java -jar MaltEval.jar -g {0} -v 1'.format(file_path))
