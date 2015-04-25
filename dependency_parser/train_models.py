import random
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
import os
from providedcode.dependencycorpusreader import DependencyCorpusReader

def main():
	scores = {}
	for lang in ['english','swedish','korean','danish']:
	#for lang in ['danish']:
		scores[lang]= train_model(lang)
	
	eval_scores(scores)

	return

def eval_scores(scores):
	total_score = 0
	for lang,score in scores.iteritems():
		if lang == 'english':
			value = 14*(min(.7,score)/.7)**2
		else:
			value = 7*(min(.7,score)/.7)**2
		print value
		total_score+=value
	log1 =  'total_score: {0}\n'.format(total_score)
	log2 =  'percentage of 35: {0}\n'.format(total_score/35)

	with open('results.txt','a') as results_file:
		results_file.write(log1)
		results_file.write(log2)
	print log1,log2
	return

# adapted datasets.py
# can pass parameters to access each language
# and each dataset
# (12 functions in one and iterative potential!)
def get_data(lang,dataset='train'):
	if dataset not in ['train','test','dev']:
		raise ValueError('dataset must be in train|test|dev')
        base_path = '/home/coms4705/Documents/Homework2/data/'
        lang_base = {'english'      :'english',
                        'korean' :'korean',
                        'danish' :'danish/ddt',
                        'swedish':'swedish/talbanken05'}

        file_name = {'english' : 'train/en-universal-train.conll',
                        'korean' : 'train/ko-universal-train.conll',
                        'danish' : 'train/danish_ddt_train.conll',
                        'swedish' : 'train/swedish_talbanken05_train.conll'}

	root = os.path.join(base_path,lang_base.get(lang))

	if dataset == 'train':
		conll_name = file_name.get(lang)
	else:
		conll_name = file_name.get(lang).replace('train',dataset)

	return DependencyCorpusReader(root,conll_name)
	
# wrapper to train a model, save and evaluate
def train_model(lang,training_set='train'):
	# load and sample data
	data = get_data(lang,dataset=training_set).parsed_sents()
	if len(data) >200:
		random.seed(1234)
		subdata = random.sample(data, 200)
	else:
		subdata = data

	# train model and save
	tp = TransitionParser(Transition, FeatureExtractor)
	tp.train(subdata)
	tp.save('{0}.model'.format(lang))


	# test performance on new data
	if lang != 'english':
		testdata = get_data(lang,dataset='test').parsed_sents()
	
	# english test data not available
	# so find a subset of training data 
	# that is disjoint from data used for training 
	else:
		not_in_training = [sent for sent in data if sent not in subdata]
		testdata = random.sample(not_in_training,200)

	parsed = tp.parse(testdata)

	ev = DependencyEvaluator(testdata, parsed)

	# store and print results
	with open('results.txt','a') as results_file:
		results_file.write('{0} model:\n'.format(lang))
		results_file.write("UAS: {} \nLAS: {}\n".format(*ev.eval()))
	print '{0} model:\n'.format(lang)
	print "UAS: {} \nLAS: {}\n".format(*ev.eval())
	return ev.eval()[1]

if __name__ == '__main__':
	main()
	pass
