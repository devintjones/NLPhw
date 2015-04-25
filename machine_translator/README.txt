Title:  NLP HW4. Alignment
Author: Devin Jones
Uni:    dj2374

# Run time
main.py runs in 7.5 minutes


# Part A

## 1 Run NLTK IBM1 and write results to ibm1.txt

## 2 Run NLTK IBM2 and write results to ibm2.txt

## 3 Compute Alignment Error Rate for the first 50 sentences

Average AER IBM1: 0.665
Average AER IBM2: 0.650

Show an example of one model outperforming the other and explain why:

For the sentence below, IBM2 outperformed IBM1:
Source sentence:[u'Das', u'war', u'der', u'Beschlu\xdf', u'.']
Target sentence:[u'That', u'was', u'the', u'decision', u'.']
True alignments: 0-0 1-1 2-2 3-3 4-4
IBM2 alignments: 0-0 1-3 2-2 3-3
IBM1 alignments: 0-1 1-0 2-2 3-0

IBM2 initialzes with IBM1 translation probabilities. After 10 additional iterations of EM using alignment probabilities not included in the IBM1 model, IBM2 was able to correctly translate high frequency tokens 'Das' and 'war'.  


In this sentence, IBM1 outperforms IBM2:
Source sentence:[u'Die', u'Aussprache', u'ist', u'geschlossen', u'.']
Target sentence:[u'The', u'debate',     u'is',  u'closed',      u'.']
True alignments: 0-0 1-1 2-2 3-3 4-4
IBM1 alignments: 0-0 1-1 2-2 3-3
IBM2 alignments: 0-0 1-3 2-2 3-3

In this case, the alignment probability for IBM2 incorrectly translated "Aussprache" to "closed" (the true translation is "debate"). This is likely because in most sentences of length 4 in German and length 4 in English, the second word most frequently translates to the 4th word. This alignment is not true in this partular case, and the more simple IBM1 model parameters were shown to be more accurate. 


## 4 Find minimum number of iterations required for AER convergence

Using epsilon = .001, AER converged for IBM1 after 11 iterations reaching an averge AER of 0.665. The program took 54 seconds to fit 11 iterations of EM. With 350 sentences, this algorithm takes approximately 5 additional seconds for each iteration. 

For IBM2, the model converged within .001 after only 3 iterations, reaching an averge AER of 0.6438. Convergence occurs in IBM2 after a small amount of iterations likeley due to the initiation of translation probabilities by 10 iterations of IBM1. If we include the number of iterations of IBM1, the total number of iterations before convergences is 13. 

For implementation, see A.converge()


# Part B

## 1 Implement BerkeleyAligner.train()

## 2 Implement BerkeleyAligner.align()

## 3 Train using BerkeleyAligner

## 4 Compute average AER for BerkeleyAligner on the first 50 sentences

Average AER Berkeley: 0.548

## 5 Give an example of a sentence pair that the Berkeley Aligner performs better than the IBM models and explain why that is the case. 

In the example below, the Berkely Aligner outperforms the IBM2 model. 
 
Source sentence:[u'Das', u'ist', u'der', u'Fall', u'von', u'Alexander', u'Nikitin', u'.']
Target sentence:[u'It', u'is', u'the', u'case', u'of', u'Alexander', u'Nikitin', u'.']
True alignments   : 0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7
Berkely alignments: 0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7
IBM2 alignments   : 0-0 1-1 2-4 3-3 4-5 5-6 6-6

Specifically, the Berkely Aligner correctly translated the following relations where the IBM2 model failed: 
"der" -> "the"
"von" -> "of"
"Alexander" -> "Alexander"

In this case, the Berkeley Aligner incorporated translation probabilities from English to German to improve alignment probability accuracy. 

Below, the Berkely Aligner outperforms IBM1:

Source sentence:[u'Das',  u'war', u'der', u'Beschlu\xdf', u'.']
Target sentence:[u'That', u'was', u'the', u'decision',    u'.']
True alignments    : 0-0 1-1 2-2 3-3 4-4
Berkeley alignments: 0-0 1-1 2-2 3-3 4-4
IBM1 alignments    : 0-1 1-0 2-2 3-0

In addition to translation probabilities from English to German incoporated into the model parameters, the Berkley Aligner accounts for index alignment probalities for same length sentences. The increased performance based in the alignment probabilitiy parameters are showcased in the correct alignment predicitons from the Berkeley vs IBM1 above.  

## 6 (EC) Modify the Berkeley Aligner to improve avg AER


