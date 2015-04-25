from nltk.corpus import comtrans
import A
import B
import EC
import time

if __name__ == '__main__':
	start = time.clock()
	aligned_sents = comtrans.aligned_sents()[:350]
	A.main(aligned_sents)
	B.main(aligned_sents)
	try:
		EC.main(aligned_sents)
	except Exception,e:
		print 'EC.py has errors:'
		print e
	print 'Processing time: {0:.3f} min'.format(time.clock()/60)
