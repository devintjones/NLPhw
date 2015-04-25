Title: Columbia University NLP HW1
Author: Devin Jones
Uni: dj2374
Date: 2-18-2015

## Runtime:
SolutionsA.py runtime: 51.65 seconds
SolutionsB.py runtime: 7.6 minutes

## Part A

### A2
* The perplexity of A2.uni.txt is 1104.83292814
* The perplexity of A2.bi.txt is 57.2215464238
* The perplexity of A2.tri.txt is 5.89521267642

### A3
* The perplexity of A3.txt is 13.0759217039

### A4
Linear interpolation lowers the perplexity of the unigram and bigram models, which can be interpreted as a more accurate language model by the lower perpeplixity, or the inverse probability of the test set. Linear interpolation creates a more robust language model in thise caseby combining information from all 3 models. 

The perplexity is still lower in the trigram model, likely because the trigram model becomes diluted by the bigram and unigram models in linear interpolation.

### A5
* The perplexity of Sample1_scored.txt is 1.55950732634
* The perplexity of Sample2_scored.txt is 7.32048000699

The sample corpus with the lower perplexity is likely from the same corpuse as the traiing data, the Brown dataset. In this case, it is 'Sample1.txt'. 

## Part B

* Percent correct tags of B5.txt (Viterbi): 84.2432435051
* Percent correct tags of B6.txt (NLTK w Backoff): 85.3033518416




