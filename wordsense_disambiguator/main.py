from xml.dom import minidom
import codecs
import sys
import unicodedata
import os
from random import shuffle

from nltk               import word_tokenize,corpus
from nltk.stem.porter   import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
import numpy as np


stopword_dict = {'English':corpus.stopwords.words('english'),
	         'Spanish':corpus.stopwords.words('spanish')}



def main():
	all_scores = []
	for language in ['English']:
		for features in ['standard_wn']:
			
			# skip invalid params 
			if language == 'Catalan' and features not in  ['standard']:
				continue
			if features == 'standard_wn' and language != 'English':
				continue
			
			for window in range(20,30,5):
				train_vecs = get_feature_vecs(language,'train',features,k=window)
				test_vecs  = get_feature_vecs(language,'dev',features,k=window) 
				#for model in ['svm','knn','reg','nb','bst','rf','ada']:
				for model in ['svm']:
					for n in range(1000,2500,500):
						topn = {'method':'pmi','n':n}
						for method in ['pmi','rel']:
							topn['method'] = method
							scores = score_params(train_vecs,test_vecs,language,
									      features,model,topn,pca=False,
									      knn=15,window=window)
							print(scores)
							all_scores.append(scores)

	return all_scores

def score_params(train_vecs,test_vecs,language,features='standard',model='svm',topn=None,pca=False,knn=5,window=10):
	results    = predict(train_vecs,test_vecs,model,topn,pca,knn)
	model_file = write_results(results,language,model,features,topn,pca,knn,window)
	score      = score_results(model_file,language)
	return (model_file,score)


# why not use json?
def get_feature_vecs(language,dataset,features='standard',k=10):
	
	#if dataset not in ['train','dev']:
	#	raise ValueError('data set must be train or dev')

	feature_lookup = {'standard'   :standard,
			  'stopwords'  :stopwords,
			  'stemmed'    :stemmed,
			  'stp_stemmed':stp_stemmed,
			  'standard_wn':standard_wn}
        
	xmldoc = minidom.parse(dataset)
	
	test_vecs = {}
	lex_list = xmldoc.getElementsByTagName('lexelt')
	for node in lex_list:
		lexelt = node.getAttribute('item')
		test_vecs[lexelt] = []

		inst_list = node.getElementsByTagName('instance')
		for inst in inst_list:
			# remove uknown senses
			if 'train' in dataset:
				sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
				if sense_id == 'U': continue

			context = inst.getElementsByTagName('context')[0]
			
			# handle the Spanish xml structure
			struct  = {node.tagName:node for node in context.childNodes if node.hasChildNodes()}
			if struct.get('target'):
				context = struct.get('target')

			pre   = feature_lookup.get(features)(context,0,language)[-k:]
			post  = feature_lookup.get(features)(context,2,language)[:k]
			s       = pre + post
			vec =  {token:s.count(token) for token in s}
			
			# get instances id and append a tuple for dev, get sense id for train
			if 'dev' in dataset:
				instance_id = inst.getAttribute('id')
				append_obj  = (instance_id, vec)
			if 'train' in dataset:
				vec['sense_id']=sense_id
				append_obj  = vec

			test_vecs[lexelt].append(append_obj)
	return test_vecs





# menu of feature generators

def standard(context,idx,language):
	tokenized = word_tokenize(context.childNodes[idx].wholeText)
	return tokenized

def stopwords(context,idx,language):
	tokens          = word_tokenize(context.childNodes[idx].wholeText)
	filtered_tokens = [w for w in tokens if not w in stopword_dict.get(language)]
	return filtered_tokens

def stemmed(context,idx,language):
	stemmer = SnowballStemmer(language.lower())
	tokens  = word_tokenize(context.childNodes[idx].wholeText)
	stemmed = [stemmer.stem(w) for w in tokens]
	return stemmed

def stp_stemmed(context,idx,language):
	stemmer         = SnowballStemmer(language.lower())	
	tokens          = word_tokenize(context.childNodes[idx].wholeText)
	filtered_tokens = [w for w in tokens if not w in stopword_dict.get(language)]
	stemmed         = [stemmer.stem(w) for w in filtered_tokens]
	return stemmed

def wordnet(tokens,language):
	to_add = []
	language_lu = {'English':'en',
		       'Spansih':'spa',
		       'Catalan':'cat'}
	for token in tokens:
		syns_list = wn.synsets(token,lang=language_lu.get(language))
		hypernyms = list(set([item.name() for obj in syns_list for item in obj.hypernyms()]))
		hyponyms  = list(set([item.name() for obj in syns_list for item in obj.hyponyms()]))
		lemmas    = list(set([item.name() for obj in syns_list for item in obj.lemmas()]))
		to_add    = to_add + hypernyms + hyponyms + lemmas
	features = tokens + to_add
	shuffle(features)
	return features

def standard_wn(context,idx,language):
	tokenized = word_tokenize(context.childNodes[idx].wholeText)
	tokenized = wordnet(tokenized,language)
	return tokenized



# Machine Learning!

def predict(train_vecs,test_vecs,method='svm',topn=None,pca=False,k=5):
	method_dict = {'svm':svm.LinearSVC(class_weight='auto'),
		       'knn':neighbors.KNeighborsClassifier(n_neighbors=k,weights='distance'),
		       'reg':linear_model.SGDClassifier(class_weight='auto'),
		       'nb' :naive_bayes.MultinomialNB(fit_prior=True),
		       'bst':ensemble.GradientBoostingClassifier(),
		       'rf' :ensemble.RandomForestClassifier(),
		       'ada':ensemble.AdaBoostClassifier()}

	feat_select = {'rel':rel,
			'pmi':pmi}

	results = {}
	for lexelt in train_vecs.keys():
		# combine test & train to get matrix mapping for all possible tokens	
		test_list = [obs[1] for obs in test_vecs.get(lexelt)]
		ids       = [obs[0] for obs in test_vecs.get(lexelt)]
		all_data  = train_vecs.get(lexelt) + test_list
		
		train_y,train_x,test_x,senses = data_pipe(all_data)

		if topn:
			train_x,test_x = feat_select.get(topn.get('method'))(test_x,train_x,train_y,topn.get('n'))
		
		if pca:
			train_x,test_x = pca_trans(train_x,test_x)
	
		# fit model
		clf       = method_dict.get(method)
		fit       = clf.fit(train_x.toarray(),train_y)
	
		# predict
		predicts        = clf.predict(test_x.toarray())
		predicted_sense = [senses[int(i)-1].split('=')[1] for i in predicts]

		# clean up & store
		records = [[ids[i], predicted_sense[i]] for i in range(len(ids))] 
		results[lexelt]=records

	return results


# dimensionality reduction

def pca_trans(train_x,test_x):
	
	transformer = PCA(n_components=.9)
	transformer.fit(train_x.toarray())
	
	new_train_x= transformer.transform(train_x.toarray())
	new_test_x = transformer.transform(test_x.toarray())
	return new_train_x,new_test_x



# select top predictive features

def rel(test_x,train_x,train_y,n=30):
	use_these = np.zeros(0)
	for sense in list(set(train_y)):
		# get all obs for sense i and obs for not sense i
		sense_idx  = [i for i in range(len(train_y)) if train_y[i] == sense]
		others     = [i for i in range(len(train_y)) if train_y[i] != sense]
		
		# get data that corresponds to above indices
		freq_sense = train_x[sense_idx,:].sum(0)
		freq_others= train_x[others,   :].sum(0)

		# compute relevancy score & convert inf to numbers
		rel_score  = np.log(np.divide(freq_sense,freq_others))
		rel_score  = np.nan_to_num(rel_score)

		# select top n features & store
		feat_select= np.asarray(np.argsort(-rel_score)[0,:n])
		use_these  = np.append(use_these,feat_select)
	# dedup indices and subset data
	use_these   = np.unique(use_these)
	new_train_x = train_x[:,use_these]
	new_test_x  =  test_x[:,use_these]
	return new_train_x,new_test_x


# point-wise mutual information
# log(p(y|x)/p(y))
def pmi(test_x,train_x,train_y,n=30):

	classes = list(set(train_y))
	
	# p(y)
	p_y = {sense:float(len(train_y[train_y==sense]))/len(train_y) for sense in classes}
	
	# idices of nonzero x values for each feature
	shape = train_x.shape
	nonzero_idx = [train_x[:,col].nonzero()[0] for col in range(shape[1])]
	
	pmi_scores = np.zeros((len(p_y),shape[1]))
	for col in range(shape[1]):

		if len(nonzero_idx[col])==0:
			continue
		# ratio of count of class given the presence of x and the size of x, all divided by p(y)
		pmi_col = [(float(sum(train_y[nonzero_idx[col]]==sense))/len(nonzero_idx[col]))/p_y.get(sense) for sense in classes]
		pmi_scores[:,col] = pmi_col

	use_these = np.zeros(0)
	for sense in range(len(classes)):
		best_features = np.argsort(-pmi_scores[sense,:])[:n]
		use_these     = np.append(use_these,best_features)
	
	use_these   = np.unique(use_these)
	new_train_x = train_x[:,use_these]
	new_test_x  =  test_x[:,use_these]
	return new_train_x,new_test_x


# Data Engineering!

def data_pipe(all_data):
	# convert dict to sparse matrix
	vectorizer= DictVectorizer()
	sp_mat    = vectorizer.fit_transform(all_data) 
	
	# find target vars
	senses    = filter(lambda x: 'sense_id=' in x,vectorizer.get_feature_names())
	sense_idx = [vectorizer.get_feature_names().index(sense) for sense in senses]
	
	# collapse y data into a categorical vector
	ordinal   = sp_mat[:,sense_idx].dot(np.diag(range(1,len(sense_idx)+1)))
	ord_vec   = ordinal.sum(1)


	# subset x data, separate test & train
	train_idx = np.nonzero(ord_vec > 0)[0]
	test_idx  = np.nonzero(ord_vec== 0)[0]
	
	cols      = sp_mat.get_shape()[1]
	feat_idx  = list(set(range(cols)) - set(sense_idx))

	train_x   = sp_mat[train_idx,:]
	test_x    = sp_mat[test_idx ,:]
	
	train_y   = ord_vec[ord_vec>0]
	train_x   = train_x[:,feat_idx]
	test_x    =  test_x[:,feat_idx]

	return train_y,train_x,test_x,senses


def replace_accented(input_str):
	nkfd_form = unicodedata.normalize('NFKD', input_str)
	return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])


def write_results(results,out_file_name):

	outfile = codecs.open(out_file_name, encoding = 'utf-8', mode = 'w')
        
	for lexelt, instances in sorted(results.iteritems(), key = lambda d: replace_accented(d[0].split('.')[0])):
		for instance_id,sid in sorted(instances,key = lambda d: int(d[0].split('.')[-1])):
			outfile.write(replace_accented(lexelt + ' ' + instance_id + ' ' + sid + '\n'))
	outfile.close()
	return out_file_name


#<hack>
def score_results(model_file,test_file,language,sense_map = ''):

	key_file = test_file.replace('xml','key')
	if language == 'English': sense_map = 'data/English.sensemap'

	out = os.popen("./scorer2 {0} {1} {2}".format(model_file,key_file,sense_map)).readlines()
	start = out[2].find('precision') + 11
	score = float(out[2][start:start+5])

	with open("scores.txt", "a") as myfile:
		myfile.write("{} {} {}\n".format(model_file,language,score))

	return score
#</hack>


feature_lu = {'English':'stp_stemmed',
		'Spanish':'stp_stemmed',
		'Catalan':'standard'}

topn_lu = {'English':{'method':'rel','n':1100},
		'Spanish':{'method':'rel','n':1400},
		'Catalan':None}

window_lu = {'English':25,
		'Spanish':25,
		'Catalan':10}
import time

if __name__ == '__main__':
        if len(sys.argv) != 5:
                print 'Usage: python main.py <training file> <test file> <outputfile> <language>'
                sys.exit(0)
	t1 = time.time()	
	training_file = sys.argv[1]
	test_file     = sys.argv[2]
	out_file_name = sys.argv[3]
	lang          = sys.argv[4]

	train_vecs = get_feature_vecs(lang,training_file,feature_lu.get(lang),k=window_lu.get(lang))
	test_vecs  = get_feature_vecs(lang,test_file,feature_lu.get(lang),k=window_lu.get(lang)) 
	
	results    = predict(train_vecs,test_vecs,'svm',topn_lu.get(lang))
	model_file = write_results(results,out_file_name)
	try:
		score      = score_results(model_file,test_file,lang)
		print out_file_name,' precision: ',score
	except:
		print 'Scoring file error'
	print '{} seconds run time'.format(time.time()-t1)
	
	pass


 
