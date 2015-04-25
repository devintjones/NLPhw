#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
import nltk
import math
from nltk.collocations import *

def calc_probabilities(brown):

        #tokenize
        tokens = [nltk.word_tokenize(sentence.lower()) for sentence in brown]
        #add sentence declarations
        for sentence in tokens:
		sentence.insert(0,'<s>')
		sentence.append('</s>')
	print tokens[0]
	#collapse list of lists into big list
	big_tokens = [word for sentence in tokens for word in sentence]
        
	#nltk collocations was really slow so I built my own frequency counters 

	#get unigram frequencies and total counts
	unigram,unigram_count = raw_freq_n(big_tokens,1)
	#calculate log probabilities
	unigram_p = calc_log_prob(unigram,unigram_count)

	#bigram freq and total counts
	bigram,bigram_count = raw_freq_n(big_tokens,2)
	#bigram probabilities
	bigram_p = calc_log_prob(bigram,bigram_count)
	
	#trigram
	trigram,trigram_count = raw_freq_n(big_tokens,3)
	#probabilities
	trigram_p = calc_log_prob(trigram,trigram_count)

	return unigram_p, bigram_p, trigram_p

#calculates raw frequencies for n-grams
def raw_freq_n(big_token_list,n):
	total_count = 0.
	ngrams = {}
	scoreable = len(big_token_list) - n + 1
	for i in range(scoreable):
		total_count+=1.
		ngram = tuple(big_token_list[i:i+n])
		if ngram not in ngrams:
			ngrams[ngram]=1
		else:
			ngrams[ngram]+=1
	return ngrams,total_count


#calculates the log probability based on a dictionary and total count
def calc_log_prob(ngram_dict,total_count):
	ngram_p = {}
	for ngram,freq in ngram_dict.iteritems():
		if freq == 0:
			ngram_p[ngram]=-1000
		else:
			ngram_p[ngram]=math.log(freq/total_count,2)
	return ngram_p 	

#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
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
    scores = []
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
    scores = []
    return scores

def main():
    #open data
    infile = open('Brown_train.txt', 'r')
    brown = infile.readlines()
    infile.close()

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

    #linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    score_output(linearscores, 'A3.txt')

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
    score_output(sample2scores, 'Sample2_scored.txt')
    '''
if __name__ == "__main__": main()
