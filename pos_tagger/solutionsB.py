import os
import nltk
import math
import time

#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
	
	#start = time.clock()
	
	'''#this is REALLY slow - 339.43 secs
	doc = [word for sentence in wbrown for word in sentence]
	word_count = {word : doc.count(word) for word in set(doc) }
	'''
	
	knownwords = []
	word_count = {}
	for sentence in wbrown:
		for word in sentence:
			#if word not in knownwords: #also slow. 17 seconds to run
				if word not in word_count:
					word_count[word] = 1
				else:
					word_count[word] +=1
					if word_count.get(word) > 4:
						knownwords.append(word)
	#deduplicate
	knownwords = list(set(knownwords)) #finshed in 0.29 secs
	
	#print 'calc_known() finished in {0} seconds'.format(time.clock()-start)
	
	return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
	rare = []
	for sentence in brown:
		new_sent = []
		for word in sentence:
			if word not in knownwords:
				new_sent.append('_RARE_')
			else:
				new_sent.append(word)
		rare.append(new_sent)
	return rare

#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')

    for sentence in rare:
        outfile.write(' '.join(sentence)[2:-1] + '\n')
    outfile.close()

#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
	#same as part A:
	trigrams = raw_freq_n(tbrown,3)
	bigrams  = raw_freq_n(tbrown,2)
	qvalues  = {ngram : math.log(trigrams.get(ngram)/bigrams.get(ngram[0:2]),2) for ngram in trigrams.keys() }
	return qvalues

#calculates raw frequencies for n-grams
def raw_freq_n(list_of_lists,n):

        ngrams = {}

	for sentence in list_of_lists:
		for i in range(n-1,len(sentence)):

			#extract ngram
			ngram = tuple(sentence[i-n+1:i+1])

			#update count of ngram
			if ngram not in ngrams:
				ngrams[ngram]=1.
			else:
				ngrams[ngram]+=1.
        return ngrams


#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output 
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word 
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):

	#flatten lists
	doc = [word for sentence in wbrown for word in sentence]
	tags= [word for sentence in tbrown for word in sentence]	

	#get unique taglist
	taglist = list(set(tags)) 

	tag_freq = { tag : tags.count(tag) for tag in taglist }

	#for each tag, find out which words occur
	emission_count = {}
	for tag in tag_freq.keys():
		tag_vals = []
		for i in range(len(doc)):
			if tag == tags[i]:
				tag_vals.append(doc[i])
		emission_count[tag] = tag_vals

	combos = {}
	for i in range(len(doc)):
		combos[(doc[i],tags[i])]=0
	combos = set(combos)
	
	evalues={}
	#for each unique word/tag combo, 
	for combo in combos:
		#calculate its emission probability
		word_list = emission_count.get(combo[1])
		count = 0.
		for word in word_list:
			if word == combo[0]:
				count += 1.
		evalues[combo]= math.log(count/tag_freq.get(combo[1]),2)
	return evalues, taglist

#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()


#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords), 
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a 
#sentence tagged in the WORD/TAG format 
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues is from the return of calc_trigrams
#evalues is from the return of calc_emissions() 
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
	tagged = [viterbi_sentence(sentence,taglist,knownwords,qvalues,evalues) for sentence in brown]
	return tagged

def viterbi_sentence(sentence,taglist,knownwords,qvalues,evalues):
	
	# replace rare words and keep track of original setnence
	orig_sentence = sentence
	for i in range(len(sentence)):
		if sentence[i] not in knownwords:
			sentence[i] = '_RARE_'

	# Initialize Viterbi Algorithm	
	viterbi     = [] # list of lists of probabilities
	backpointer = [] # keeps track of the POS sequences, corresponds to above
	# Start at 3rd object to account for start symbols
	for s in range(len(taglist)):
		# these will return -1000 if the tuple isn't found in the dict
		qvalue = qvalues.get(tuple([sentence[0],sentence[1],taglist[s]]),-1000) #a_i,j
		evalue = evalues.get(tuple([sentence[2],taglist[s]]),-1000)             #b_j(o_t)
		viterbi.append([qvalue + evalue])
		backpointer.append(['*','*',taglist[s]])

	# Iterate over words and sequences
	# starting at the second word and ending at the second to last word
	for t in range(3,len(sentence)):
		# Extract viterbi probabilities from time t-1
		last_state_probs = [row[t-3] for row in viterbi]
	
		#compute probabilities across all possible states
		for s in range(len(taglist)):
			v_update = []
			bp_update = []
			tuples = []
			# iterate over all backpointer values for each state
			for s_prime in range(len(taglist)):
				qvalue = qvalues.get(tuple([backpointer[s_prime][t-2],backpointer[s_prime][t-1],taglist[s]]),-1000.) 	
				tuples.append(tuple([backpointer[s_prime][t-2],backpointer[s_prime][t-1],taglist[s]]))
				# check max of previous state to this state
				eval_last = evalues.get(tuple([sentence[t],taglist[s]]),-1000.)
				
				#handle termination step
				if t==len(sentence):
					v_update.append(last_state_probs[s_prime]+eval_last)
				else:
					v_update.append(last_state_probs[s_prime]+eval_last+qvalue)
				bp_update.append(last_state_probs[s_prime]+eval_last)
			#find the best state from computed probabilities
			viterbi[s].append(           max(v_update ))
			#select best backpointer value
			best_state = bp_update.index(max(bp_update))
			#update backpointer
			backpointer[s][-2] = tuples[best_state][0]
			backpointer[s][-1] = tuples[best_state][1]
			backpointer[s].append(tuples[best_state][2])
	
	#backtrace
	#aggregate all of the last probability entries in viterbi
	start_backtrace = [entry[-1] for entry in viterbi]
	startval = start_backtrace.index(max(start_backtrace))
	path_selection = backpointer[startval]

	#build word/tag list
	tagged_sent = []
	for i in range(2,len(sentence)-1):
		tagged_sent.append('/'.join([orig_sentence[i],path_selection[i]]))
	final_sent = ' '.join(tagged_sent)
	return final_sent

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence + '\n')
    outfile.close()

#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of tagged sentences. Each sentence is in the WORD/TAG format and is a string rather than a list of tokens.
def nltk_tagger(brown_dev):

	# import data
	from nltk.corpus import brown
	training = brown.tagged_sents(tagset='universal')

	# build taggers based on training data
	default_tagger = nltk.DefaultTagger('NN')
	unigram_tagger = nltk.UnigramTagger(training, backoff=default_tagger)
	bigram_tagger  = nltk.BigramTagger( training, backoff=unigram_tagger)
	trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

	# tag test set & clean data for output
	tagged = []
	for sentence in brown_dev:
		tagged_sent = trigram_tagger.tag(sentence[2:-1]) #skip sentence start/stop markers
		clean_tagged=['/'.join(word) for word in tagged_sent]
		tagged.append(' '.join(clean_tagged))
	return tagged

def q6_output(tagged):
    outfile = open('B6.txt', 'w')

    for sentence in tagged:
        outfile.write(sentence + '\n')
    outfile.close()

#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only) 
def split_wordtags(brown_train):
	wbrown = []
	tbrown = []

        for sentence in brown_train:
		sent_words = []
		sent_tags  = []
                sentence   = sentence.split()
		sentence.insert(0,'*/*')
		sentence.insert(0,'*/*')
                sentence.append('STOP/STOP')
		for word in sentence:
			parts = word.split('/')
			sent_words.append(''.join(parts[0:-1])) #handle a words that contain '/'
			sent_tags.append(parts[-1])	        #assumes tags don't have '/'
		wbrown.append(sent_words)
		tbrown.append(sent_tags)
	return wbrown, tbrown

def quickload():

	#open Brown training data
	infile = open("Brown_tagged_train.txt", "r")
	brown_train = infile.readlines()
	infile.close()

	#split words and tags, and add start and stop symbols (question 1)
	wbrown, tbrown = split_wordtags(brown_train)
	   
	#calculate trigram probabilities (question 2)
	qvalues = calc_trigrams(tbrown)

	#question 2 output
	q2_output(qvalues)

	#calculate list of words with count > 5 (question 3)
	knownwords = calc_known(wbrown)

	#get a version of wbrown with rare words replace with '_RARE_' (question 3)
	wbrown_rare = replace_rare(wbrown, knownwords)

	#question 3 output
	q3_output(wbrown_rare)

	#calculate emission probabilities (question 4)
	#this takes a few minutes
	evalues, taglist = calc_emission(wbrown_rare, tbrown)

	#question 4 output
	q4_output(evalues)

	#delete unneceessary data
	del brown_train
	del wbrown
	del tbrown
	del wbrown_rare

	#open Brown development data (question 5)
	infile = open("Brown_dev.txt", "r")
	brown_dev = infile.readlines()
	infile.close()

	def clean_sentence(sentence):
		sentence   = nltk.word_tokenize(sentence)
		sentence.insert(0,'*')
		sentence.insert(0,'*')
		sentence.append('STOP')
		return sentence
	
	brown_dev = [clean_sentence(sentence) for sentence in brown_dev]
	sentence=brown_dev[0]
	return sentence,brown_dev,knownwords,taglist,qvalues,evalues

def main():
	start = time.clock()

	#open Brown training data
	infile = open("Brown_tagged_train.txt", "r")
	brown_train = infile.readlines()
	infile.close()

	#split words and tags, and add start and stop symbols (question 1)
	wbrown, tbrown = split_wordtags(brown_train)
	   
	#calculate trigram probabilities (question 2)
	qvalues = calc_trigrams(tbrown)

	#question 2 output
	q2_output(qvalues)

	#calculate list of words with count > 5 (question 3)
	knownwords = calc_known(wbrown)

	#get a version of wbrown with rare words replace with '_RARE_' (question 3)
	wbrown_rare = replace_rare(wbrown, knownwords)

	#question 3 output
	q3_output(wbrown_rare)

	#calculate emission probabilities (question 4)
	#this takes a few minutes
	evalues, taglist = calc_emission(wbrown_rare, tbrown)

	#question 4 output
	q4_output(evalues)

	#delete unneceessary data
	del brown_train
	del wbrown
	del tbrown
	del wbrown_rare

	#open Brown development data (question 5)
	infile = open("Brown_dev.txt", "r")
	brown_dev = infile.readlines()
	infile.close()

	#format brown development data
	def clean_sentence(sentence):
		sentence   = nltk.word_tokenize(sentence)
		sentence.insert(0,'*')
		sentence.insert(0,'*')
		sentence.append('STOP')
		return sentence
	
	brown_dev = [clean_sentence(sentence) for sentence in brown_dev]

	#do viterbi on brown_dev (question 5)
	viterbi_tagged = viterbi(brown_dev, taglist, knownwords, qvalues, evalues)

	#question 5 output
	q5_output(viterbi_tagged)
	os.system('python pos.py B5.txt Brown_tagged_dev.txt >> README.txt')

	#do nltk tagging here
	nltk_tagged = nltk_tagger(brown_dev)

	#question 6 output
	q6_output(nltk_tagged)
	os.system('python pos.py B6.txt Brown_tagged_dev.txt >> README.txt')

	print 'SolutionsB.py runtime: {0} minutes'.format((time.clock()-start)/60)
if __name__ == "__main__": main()
