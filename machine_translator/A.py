import nltk
from nltk.corpus import comtrans
from nltk.align  import IBMModel1
from nltk.align  import IBMModel2


# Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents,n=10):
	ibm1 = IBMModel1(aligned_sents,n)	
	return ibm1


# Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents,n=10):
	ibm2 = IBMModel2(aligned_sents,n)	
	return ibm2


# tune the number of iterations based on a small change in avg AER across models
def converge(aligned_sents,model='ibm1',min_its=2,max_its=50,eps=.001):
	progress = 1
	last_err = 1
	n        = min_its
	models = {'ibm1':create_ibm1,'ibm2':create_ibm2}

	while progress > eps and n < max_its:
		#start = time.clock()

		print 'fitting {} n={}'.format(model,n)
		model_obj = models.get(model)(aligned_sents,n)
		
		avg_err = compute_avg_aer(aligned_sents,model_obj)
		print 'avg_err={}, processing time: NULL seconds'.format(avg_err)
	
		progress = abs(last_err-avg_err)
		print 'progress: {}'.format(progress)
		
		last_err = avg_err
		n+=1

	if n==max_its:
		print 'model failed to converge within {} after {} iterations.'.format(eps,max_its)
	else:
		print 'model convergence at {} iterations'.format(n-1)
	return


# : Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n=50):
	total = 0
	for i in range(n):
		total += model.align(aligned_sents[i]).alignment_error_rate(aligned_sents[i].alignment)
	return total/n


# : Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
	f = open(file_name,'w')
	for i in range(20):
		pred = model.align(aligned_sents[i])
		f.write(' '.join(pred.words).encode('utf-8') + '\n' )
		f.write(' '.join(pred.mots).encode('utf-8') + '\n')
		f.write(str(pred.alignment) + '\n\n')
	f.close()
	return



def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
