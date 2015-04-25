#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
import nltk
import math
from nltk.collocations import *
import sys
import os

def calc_probabilities(brown):

        tokens = clean_text(brown)

        #collapse list of lists into big list
        big_tokens = [word for sentence in tokens for word in sentence]

        #nltk collocations was really slow so I built my own frequency counter and probability calculator 

        #get unigram frequencies and total counts
        unigram,unigram_count = raw_freq_n(big_tokens,1)
        #calculate log conditional probabilities
        unigram_p = calc_log_prob(unigram,unigram_count)

        #bigram freq and total counts
        bigram,bigram_count = raw_freq_n(big_tokens,2)
        #bigram probabilities
        bigram_p = calc_log_prob(bigram,bigram_count,
                                 unigram,unigram_count)

        #trigram
        trigram,trigram_count = raw_freq_n(big_tokens,3)
        #probabilities
        trigram_p = calc_log_prob(trigram,trigram_count,
                                  bigram,bigram_count)

        return unigram_p, bigram_p, trigram_p




#	return unigram_p, bigram_p, trigram_p

#this is used in calc_probabilities() and score()
def clean_text(brown):
        #tokenize and lowercase
        tokens = [nltk.word_tokenize(sentence.lower()) for sentence in brown]
        #add sentence declarations
        for sentence in tokens:
                sentence.insert(0,'*')
		sentence.insert(0,'*')
                sentence.append('STOP')
	return tokens

#calculates raw frequencies for n-grams
def raw_freq_n(big_token_list,n):
	
	#init count and return data structure
	total_count = 0.
	ngrams = {}
	
	#skip end of list for chunks smaller than n
	scoreable = len(big_token_list) - n + 1

	for i in range(scoreable):
		
		#extract ngram
		ngram = tuple(big_token_list[i:i+n])
		
		#skip ngrams that strattle sentences
		if '*' in ngram and ngram[0]!='*': #'*' must be in first position
			pass
		if 'STOP' in ngram and ngram[len(ngram)-1]!='STOP': #stop must be in last position
			pass
	

		#pass wasn't working so had to do an if/else 
		else:
			#update token count
			total_count+=1.
			
			#update count of ngram
			if ngram not in ngrams:
				ngrams[ngram]=1
			else:
				ngrams[ngram]+=1
	return ngrams,total_count


#calculates the log conditional probability of the last word of an ngram
#given an ngram and the n-1 gram dicionaries
def calc_log_prob(ngram_dict,total_count_n,
		  n_1_gram_dict={},total_count_n_1=0.):

	ngram_p = {}
	
	#the unigram case
	if len(ngram_dict.keys()[0])==1:

		for ngram,freq in ngram_dict.iteritems():	
			exclude_these = [('*',)]
			if ngram in exclude_these:
				pass
			else:
				if freq == 0:
					ngram_p[ngram]=-1000
				else:
					ngram_p[ngram]=math.log(freq/total_count_n,2)
	#the n>1 case
	else:
		if n_1_gram_dict=={}:
			sys.exit('Error: Must supply n-1 gram frequency dictionary')

		for ngram,freq in ngram_dict.iteritems():
			if ngram == ('*','*'):
				pass
			else:
				#freq should be >=1, but just in case i'll hanle the log computation errors:
				if freq == 0:
					ngram_p[ngram]=-1000
				else:
					#In P(A|B), b is the first n-1 words of the n gram
					b = ngram[0:len(ngram)-1]
					#Use the n-1 gram dict to retrive those counts
					b_count = n_1_gram_dict.get(b)
					#Finally, the difference of log probabilities is equal to the log of the ratio of probabilities
					#Thus, we have log(P(A intersect B)/P(B))
					try:
						ngram_p[ngram]=math.log(freq/total_count_n,2)-math.log(b_count/total_count_n_1,2) 
					except:
						print 'ngram missing from vocabulary: {0},{1}'.format(ngram,b)
			
	return ngram_p 	

#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
	if unigram not in [('*',)]:
            outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
	if bigram not in [('*','*')]:
            outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    outfile.close()
    
#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):

	data = clean_text(data)
	scores = []
	exclude_these = [('*',),('*','*')]
	
	for sentence in data:
		sent_score = 0
		scoreable = len(sentence) - n + 1
		for i in range(scoreable):
			gram = tuple(sentence[i:i+n])
			#don't score sentence starters
			if gram not in exclude_these:
				try:
					sent_score += ngram_p.get(gram)
				except:
					print 'ngram not found in vocabulary: {0}'.format(gram)
		scores.append(sent_score)

	return scores


#this function outputs the score output of score()
#scores is a python list of scores, and filename is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


#this function scores brown data with a linearly interpolated model
#each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
#like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, brown):
	
	data = clean_text(brown)	
	scores = []
	lambdA = 1/3.

	for sentence in data:
		sent_score = 0
		for i in range(2,len(sentence)):
			
			#fetch scores from each of the models for each word
			uni_score = fetch_prob(i,1,sentence,unigrams)
			bi_score  = fetch_prob(i,2,sentence,bigrams)
			tri_score = fetch_prob(i,3,sentence,trigrams)
			
			#store in list for easy manipulation
			word_scores = [uni_score,bi_score,tri_score]

			#aggregate the score of the sentence by averaging the valid models
			sent_score += sum( [lambdA*score for score in word_scores] )

		scores.append(sent_score)
	return scores

def fetch_prob(i,n,sentence,ngram_p):
	n_score = 0
	gram = tuple(sentence[i-n+1:i+1])
	try:
		n_score += ngram_p.get(gram)
	except:
		print 'ngram not found in vocabulary: {0}. n={1}'.format(gram,n)	
	return n_score
	

#def main():
#open data
infile = open('Brown_train.txt', 'r')
brown = infile.readlines()
infile.close()

tokens = clean_text(brown)


#calculate ngram probabilities (question 1)
unigrams, bigrams, trigrams = calc_probabilities(brown)
'''
#question 1 output
q1_output(unigrams, bigrams, trigrams)

#score sentences (question 2)
uniscores = score(unigrams, 1, brown)
biscores = score(bigrams, 2, brown)
triscores = score(trigrams, 3, brown)

#question 2 output
score_output(uniscores, 'A2.uni.txt')
score_output(biscores, 'A2.bi.txt')
score_output(triscores, 'A2.tri.txt')

for file_name in ['A2.uni.txt','A2.bi.txt','A2.tri.txt']:
os.system('python perplexity.py '+file_name+' Brown_train.txt >> README.txt')
'''
#linear interpolation (question 3)
linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    #score_output(linearscores, 'A3.txt')
    #os.system('python perplexity.py A3.txt Brown_train.txt >> README.txt')
    
'''
#open Sample1 and Sample2 (question 5)
infile = open('Sample1.txt', 'r')
sample1 = infile.readlines()
infile.close()
infile = open('Sample2.txt', 'r')
sample2 = infile.readlines()
infile.close() 

#score the samples
sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

#question 5 output
score_output(sample1scores, 'Sample1_scored.txt')
os.system('python perplexity.py Sample1_scored.txt Brown_train.txt >> README.txt')
score_output(sample2scores, 'Sample2_scored.txt')
os.system('python perplexity.py Sample2_scored.txt Brown_train.txt >> README.txt')
'''
#if __name__ == "__main__": main()
