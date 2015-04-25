#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
import nltk
import math
import sys
import os
import time

def calc_probabilities(brown):
	
	#tokenize and add start/stop symbols for trigram model
	tokens = clean_text(brown)

	#get n-gram probabilities
	unigram_p = calc_ngram(tokens,1)
	bigram_p  = calc_ngram(tokens,2)
	trigram_p = calc_ngram(tokens,3)

	return unigram_p, bigram_p, trigram_p

#this is used in calc_probabilities() and score()
def clean_text(brown):
        #tokenize and lowercase
        tokens = [nltk.word_tokenize(sentence) for sentence in brown]
        #add sentence start/stop markers
        for sentence in tokens:
                sentence.insert(0,'*')
		sentence.insert(0,'*')
                sentence.append('STOP')
	return tokens

#calculates probabilities of ngram. for n>1, calculates conditional probability
def calc_ngram(tbrown,n):
        n_grams   = raw_freq_n(tbrown,n)
	if n > 1:
		n_1_gram  = raw_freq_n(tbrown,n-1,conditional=True)
		probs  = {ngram : math.log(n_grams.get(ngram)/n_1_gram.get(ngram[:n-1]),2) for ngram in n_grams.keys() }
        else:
		total_count = sum(n_grams.values())
		probs = {ngram : math.log(n_grams.get(ngram)/total_count,2) for ngram in n_grams.keys() }
	return probs

#calculates raw frequencies for n-grams
def raw_freq_n(list_of_lists,n,conditional=False):
        ngrams = {}

	# can specify whether we are building values for the denomintor of a 
	# conditoinal probability computation
	start=2
	if conditional:start=1
        
	for sentence in list_of_lists:
                for i in range(start,len(sentence)): 
                        #extract ngram
                        ngram = tuple(sentence[i-n+1:i+1])
                        #update count of ngram
                        if ngram not in ngrams:
                                ngrams[ngram]=1.
                        else:
                                ngrams[ngram]+=1.
        return ngrams


#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
	try:
            outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
	except:
	    print 'unigram out of range: {0}'.format(unigram)
    for bigram in bigrams:
	    try:
        	outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
	    except:
		print 'bigram out of range: {0}'.format(bigram)
    for trigram in trigrams:
	    try:
        	outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
	    except:
		print 'trigram out of range: {0}'.format(trigram)
		
    outfile.close()
    
#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):

	data = clean_text(data)
	scores = []	
	for sentence in data:
		sent_score = 0
		for i in range(2,len(sentence)):
			#extract ngram based on n, the final word position, 
			#and the sentence
			gram = tuple(sentence[i-n+1:i+1])
			word_score =  ngram_p.get(gram)
			
			#handle out of dictionary words
			if word_score is None:
				sent_score = -1000
				#print 'ngram not found in vocabulary: {0}'.format(gram)
				break
			else:
				sent_score += word_score

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
	# tokenize and add start/stop symbols	
	data = clean_text(brown)	
	scores = []
	lambdA = 1/3.

	for sentence in data:
		sent_score = 0
		#start at 2 to skip sentence start markers for trigram formatted corpus
		for i in range(2,len(sentence)):
			
			#fetch scores from each of the models for each word
			uni_score = fetch_prob(i,1,sentence,unigrams)
			bi_score  = fetch_prob(i,2,sentence,bigrams)
			tri_score = fetch_prob(i,3,sentence,trigrams)
			
			#handle unknown
			if uni_score == None or bi_score == None or tri_score ==None:
				sent_score = -1000
				break
			else:
				#combine the log probabilities
				word_score = math.log(lambdA *(2**uni_score + 2**bi_score + 2**tri_score),2)


				#aggregate the score of the sentence by averaging the valid models
				sent_score += word_score

		scores.append(sent_score)
	return scores

def fetch_prob(i,n,sentence,ngram_p):
	# build n gram based on the sentence, word postion and n
	gram = tuple(sentence[i-n+1:i+1])
	
	# will return None if not found
	n_score = ngram_p.get(gram)
	return n_score
	

def main():
    start = time.clock()
    #open data
    infile = open('Brown_train.txt', 'r')
    brown = infile.readlines()
    infile.close()

    #calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(brown)
    
    #question 1 output
    q1_output(unigrams, bigrams, trigrams)
    
    #score sentences (question 2)
    uniscores = score(unigrams, 1, brown)
    biscores  = score(bigrams, 2, brown)
    triscores = score(trigrams, 3, brown)

    #question 2 output
    score_output(uniscores, 'A2.uni.txt')
    score_output(biscores, 'A2.bi.txt')
    score_output(triscores, 'A2.tri.txt')
    
    #this writes the scores to the readme
    
    for file_name in ['A2.uni.txt','A2.bi.txt','A2.tri.txt']:
        os.system('python perplexity.py '+file_name+' Brown_train.txt >> README.txt')
    
    #linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    score_output(linearscores, 'A3.txt')
    #write score to readme
    os.system('python perplexity.py A3.txt Brown_train.txt >> README.txt')
    
    
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
    
    print 'SolutionsA.py runtime: {0} seconds'.format(time.clock()-start)
if __name__ == "__main__": main()
