from svm_knn import *

language='English'
dataset='train'
features='standard'

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

feature_lookup = {'standard'   :standard,
		  'stopwords'  :stopwords,
		  'stemmed'    :stemmed,
		  'stp_stemmed':stp_stemmed,
		  'standard_wn':standard_wn}

xmldoc = minidom.parse('data/{}-{}.xml'.format(language,dataset))

test_vecs = {}
lex_list = xmldoc.getElementsByTagName('lexelt')
node = lex_list[0]
lexelt = node.getAttribute('item')
test_vecs[lexelt] = []

inst_list = node.getElementsByTagName('instance')
inst = inst_list[0]

if dataset == 'train':
	sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')


context = inst.getElementsByTagName('context')[0]

# handle the Spanish xml structure
struct  = {node.tagName:node for node in context.childNodes if node.hasChildNodes()}
if struct.get('target'):
	context = struct.get('target')
k=25
pre   = feature_lookup.get(features)(context,0,language)[-k:]
post  = feature_lookup.get(features)(context,2,language)[:k]
s       = pre + post
vec =  {token:s.count(token) for token in s}

# get instances id and append a tuple for dev, get sense id for train
if dataset == 'dev':
	instance_id = inst.getAttribute('id')
	append_obj  = (instance_id, vec)
if dataset == 'train':
	vec['sense_id']=sense_id
	append_obj  = vec

window=25
train_vecs = get_feature_vecs(language,'train',features,k=window)
test_vecs  = get_feature_vecs(language,'dev',features,k=window)

test_list = [obs[1] for obs in test_vecs.get(lexelt)]
ids       = [obs[0] for obs in test_vecs.get(lexelt)]
all_data  = train_vecs.get(lexelt) + test_list

train_y,train_x,test_x,senses = data_pipe(all_data)

