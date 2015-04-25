import A
import B
from nltk.corpus import comtrans


aligned_sents = comtrans.aligned_sents()[:350]

ibm1 = A.create_ibm1(aligned_sents)
ibm2 = A.create_ibm2(aligned_sents)
berkeley = B.BerkeleyAligner(aligned_sents,10)

def compare_models(model1,model2,alignSent):
	score1 = model1.align(alignSent).alignment_error_rate(alignSent.alignment)
	score2 = model2.align(alignSent).alignment_error_rate(alignSent.alignment)
	return score2-score1

def compare_sent(model1,model2,aligned_sents):
	max_diff = 0
	for alignSent in aligned_sents:
		diff = compare_models(model1,model2,alignSent)
		if diff > max_diff:
			max_diff  = diff
			diff_sent = alignSent
	print 'Source sentence:' + str(diff_sent.words)
	print 'Target sentence:' + str(diff_sent.mots)
	print 'True alignments:   ' + str(diff_sent.alignment)
	print 'model1 alignments: ' + str(model1.align(diff_sent).alignment)
	print 'model2 alignments: ' + str(model2.align(diff_sent).alignment)

compare_sent(berkeley,ibm2,aligned_sents)
