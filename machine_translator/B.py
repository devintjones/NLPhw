from __future__ import division
import nltk
import A
from nltk.align.ibm1 import IBMModel1
from nltk.align  import AlignedSent
from collections import defaultdict



class BerkeleyAligner():

	def __init__(self, align_sents, num_iter):
		self.t, self.q = self.train(align_sents, num_iter)
		return 

	# Computes the alignments for align_sent, using this model's parameters. Return
	#       an AlignedSent object, with the sentence pair and the alignments computed.
	def align(self, align_sent):
		alignment = []

        	l_e = len(align_sent.words)
        	l_f = len(align_sent.mots)
		
	        for j, en_word in enumerate(align_sent.words):
            
			# Initialize the maximum probability with Null token
			max_align_prob = (self.t[en_word][None]*self.q[0][j+1][l_e][l_f], None)
		
			for i, fr_word in enumerate(align_sent.mots):
				# Find out the maximum probability
				max_align_prob = max(max_align_prob,
				    (self.t[en_word][fr_word]*self.q[i+1][j+1][l_e][l_f], i))

			if max_align_prob[1] is not None:
				alignment.append((j, max_align_prob[1]))

		return AlignedSent(align_sent.words, align_sent.mots, alignment)	


	# Implement the EM algorithm. num_iters is the number of iterations. Returns the 
	# translation and distortion parameters as a tuple.
	def train(self, align_sents, num_iters):

		    
		# Initialize the distribution of alignment probability,
		# a(i|j,l_e, l_f) = 1/(l_f + 1)
		def init_align(align_sents):
			align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float))))
			for alignSent in align_sents:
				en_set = [None] + alignSent.words
				fr_set = [None] + alignSent.mots
				l_f = len(fr_set) - 1
				l_e = len(en_set) - 1
				initial_value = 1 / (l_f + 1)
				for i in range(0, l_f+1):
					for j in range(0, l_e+1):
						align[i][j][l_e][l_f] = initial_value
			return align


		def get_counts(align_sents,align,t_ef):
			count_ef = defaultdict(lambda: defaultdict(float))
			total_f = defaultdict(float)

			count_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
			total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

			total_e = defaultdict(float)

			for alignSent in align_sents:
				en_set = [None] + alignSent.words
				fr_set = [None] + alignSent.mots
				l_f = len(fr_set) - 1
				l_e = len(en_set) - 1

				# compute normalization
				for j in range(0, l_e+1):
					en_word = en_set[j]
					total_e[en_word] = 0
					for i in range(0, l_f+1):
						total_e[en_word] += t_ef[en_word][fr_set[i]] * align[i][j][l_e][l_f]

				# collect counts
				for j in range(0, l_e+1):
					en_word = en_set[j]
					for i in range(0, l_f+1):
						fr_word = fr_set[i]
						c = t_ef[en_word][fr_word] * align[i][j][l_e][l_f] / total_e[en_word]
						count_ef[en_word][fr_word] += c
						total_f[fr_word] += c
						count_align[i][j][l_e][l_f] += c
						total_align[j][l_e][l_f] += c
			return count_ef,total_f,count_align,total_align
	


		def compute_align(align_sents,align,count_align,total_align):	
			# Estimate the new alignment probabilities
			for alignSent in align_sents:
				en_set = [None] + alignSent.words
				fr_set = [None] + alignSent.mots
				l_f = len(fr_set) - 1
				l_e = len(en_set) - 1
				for i in range(0, l_f+1):
					for j in range(0, l_e+1):
						align[i][j][l_e][l_f] = count_align[i][j][l_e][l_f] / total_align[j][l_e][l_f]
			return align

		
		# Initialize data structures and probability estimates
		fr_vocab = set()
		en_vocab = set()
		for alignSent in align_sents:
			en_vocab.update(alignSent.words)
			fr_vocab.update(alignSent.mots)
		fr_vocab.add(None)
		en_vocab.add(None)
				
		inverted = [alignSent.invert() for alignSent in align_sents]
		align_en = init_align(align_sents)
		align_fr = init_align(inverted)

		init_ef = 1./len(fr_vocab)
		init_fe = 1./len(en_vocab)
		t_ef = defaultdict(lambda: defaultdict(lambda: init_ef))
		t_fe = defaultdict(lambda: defaultdict(lambda: init_fe))
		
		# iterate expected count and alignment probability computation
		for i in range(0, num_iters):
			count_ef,total_f,count_align_ef,total_align_f = get_counts(align_sents,align_en,t_ef)
			count_fe,total_e,count_align_fe,total_align_e = get_counts(inverted,   align_fr,t_fe)
			
			# update index alignment probabilities
			align_en = compute_align(align_sents,align_en,count_align_ef,total_align_f)
			align_fr = compute_align(inverted,   align_fr,count_align_fe,total_align_e)

			# combined objective
			t_ef = t_fe = defaultdict(lambda: defaultdict(lambda: 0.0))
			for f in fr_vocab:
				for e in en_vocab:
					t_ef[e][f] = t_fe[f][e] = ( count_ef[e][f] + count_fe[f][e] ) / ( total_f[f]+ total_e[e] )

		# probabilties used for alignment prediction:
		t = t_ef

		# combine index align probabilities for future alignment predicition
		q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
		total = 0
		bad = 0
		for alignSent in align_sents:

			en_set = [None] + alignSent.words
			fr_set = [None] + alignSent.mots
			l_f = len(fr_set) - 1
			l_e = len(en_set) - 1
			for i in range(0, l_f+1):
				for j in range(0, l_e+1):
					total +=1
					try:
						q[i][j][l_e][l_f] = align_en[i][j][l_e][l_f] * align_fr[j][i][l_f][l_e]
					except:
						print i,j,align_en[i][j][l_e][l_f],align_fr[j][i][l_f][l_e]
						bad+=1
		if bad > 0: print 'total: {},bad: {}, %: {}'.format(total,bad,float(bad)/total)

		return (t,q)

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 20)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
