author: Devin Jones
uni: dj2374

# NLP HW3 Report

## Q2 Compare the performance of KNN & SVM

SVM outperformed KNN in all language cases with no modifications to any of the model parameters or feature vectors. 

The default parameters were: context vector window size=10, no token modification, for SVM C=1, and for KNN neighbors = 5.

The precision from each of the above mentioned models are as follows:

Language |  SVM  |  KNN  | Basline
---------|-------|-------|---------
English  | 0.623 | 0.568 |  0.535
Spanish  | 0.783 | 0.698 |  0.684
Catalan  | 0.824 | 0.708 |  0.678

For English, SVM yielded a 16% improvement in precision over the baseline, for Spanish 14%, and for Catalan 21.5%.
Comparatively, KNN yielded the following lifts from KNN respectively: 6%, 2%, and 4%.
SVM is the better predictor by a large margin. 


## Q4 Discuss how each feature effects predictor performance

### Q4A Stopwords and Stemming
#### English stopwords & stemming results:

Feature              |  SVM  | KNN
---------------------|-------|-----
standard	     | 0.623 | 0.568
stopwords            | 0.618 | 0.538
stemmed              | 0.619 | 0.559
stopwords & stemmed  | 0.627 | 0.562

Stopwords and stemming by themselves did not increase performance of either SVM or KNN. However, when they were both implemented on the conext vectors, the performance of SVM increased marginally over the unmanipulated context vector (0.6%) and 17% over the baseline.

#### Spanish stopwords & stemming

Feature              |  SVM  | KNN
---------------------|-------|-----
standard	     | 0.783 | 0.698
stopwords            | 0.785 | 0.691
stemmed              | 0.795 | 0.702
stopwords & stemmed  | 0.800 | 0.708

For the Spanish dataset, word sense disambiguation improved sequentially as the context vector was cleaned by removing stopwords, stemming, and finally by removing stopwords and stemming at the same time. The top performer was SVM with stemming and stopwords removed, improving perfmance over the standard feature set by 2% and 17% over the baseline. 

#### Cataline stopwrods and stemming
NLTK does not support stopwords and stemming for stopwords and stemming. 


### Q4B Wordnet synonyms, hyponyms, and hypernyms

The NLTK wordnet API is not available for either Spanish or Catalan. Adding these features was shown to lower precision below the baseline, likely because these features were redundant and added noise to the training data. 

Language |  SVM  |  KNN  | Basline
---------|-------|-------|---------
English  | 0.529 | 0.484 |  0.535


### Q4C Relevance Score

The relevance score was shown to marginally improve the performance of the word sense disambiguation model under certain circumstances after tuning the number of features added to the feature set as well as the context vector window size. 

Other classifiers were tested including Naive Bayes, Regularized Regression, Gradient Boosted Trees, Random Forests, and Ada Boost. Naive Bayes demonstrated strong performance which can be attributed to the model's appropriate handling of sparse, discrete data. 
 
Top performing results of a grid search on model, feature type, window size and number of features are shown below. 


Language |  Model | Feature Type | # Feats | Window Size | Precision | SVM Precision (no feature selection)
---------|--------|--------------|---------|-------------|-----------|---------
English  |  SVM   | stp-stemmed  |  1200   |    10       | 0.627     | 0.627
English  |  NB    | stp-stemmed  |  1400   |    20       | 0.624     | 0.627
English  |  SVM   | standard     |  1200   |    10       | 0.623     | 0.623

Spanish  |  NB    | stp-stemmed  |  1300   |    25       | 0.817     | 0.800
Spanish  |  SVM   | stp-stemmed  |  1200   |    10       | 0.800     | 0.800

Catalan  |  SVM   | standard     |  1200   |    10       | 0.824     | 0.824

The relevance score feature selection appears to only improve the naive bayes Spanish language model. 


### Q4D Other Feature Selection Methods

Point wise mutual information was one method used to select the most relevant features. When tuning for number of features and window size, the maximum precision improvement over no feature selection was 1.7%, 3.6% and no change for English, Spanish and Catalan respectively.

 
Language |  Model | Feature Type | # Feats | Window Size | Precision | SVM Precision (no feature selection)
---------|--------|--------------|---------|-------------|-----------|---------
English  |  SVM   | stp-stemmed  |  1200   |    20       | 0.638     | 0.627
English  |  NB    | stp-stemmed  |  1400   |    25       | 0.633     | 0.627

Spanish  |  SVM   | stp-stemmed  |  1500   |    25       | 0.829     | 0.800
Spanish  |  NB    | stp-stemmed  |  1200   |    20       | 0.827     | 0.800

Catalan  |  SVM   | standard     |  1200   |    10       | 0.824     | 0.824



I also experimented with PCA which did not improve precision over SVM with no feature selection. 


Language |  Model | Feature Type | # Feats | Window Size | Precision | SVM Precision (no feature selection)
---------|--------|--------------|---------|-------------|-----------|---------
English  |  SVM   | wordnet      |  NA     |    10       | 0.616     | 0.627
English  |  SVM   | stp-stemmed  |  NA     |    10       | 0.615     | 0.627

Spanish  |  SVM   | stp-stemmed  |  NA     |    10       | 0.763     | 0.800



### Q5 Best results

KNN nearest neighbors parameter was tested over 5,10,and 15 and over weighting by distance and uniform. None of these paramters, coupled with window size and feature selection improved precision over SVM. Additionally, ignoring capitalization and punctation was shown to not improve performance. Over 1100 combinations of model types and parameter settings were tested overall. 

Significant performance gains were realized after setting the SVM C parameter to 'auto' which balances C according to each class model, setting the value equal to the inverse proportion of frequency in the training set. Overall precision improvements over English, Spanish and Catalan were 19%, 21% and 21% over baseline respectively after grid searching for optimal parameter settings for number of features and window size.  


Language |  Model | Feature Type | Feat Sel| # Feats | Window Size | Precision | Baseline
---------|--------|--------------|---------|---------|-------------|-----------|---------
English  |  SVM   | stp-stemmed  |  rel    |   1100  |    20       | 0.639     | 0.535
English  |  SVM   | stp-stemmed  |  rel    |   1100  |    25       | 0.639     | 0.535
English  |  SVM   | stp-stemmed  |  pmi    |   1200  |    20       | 0.638     | 0.535

Spanish  |  SVM   | stp-stemmed  |  rel    |   1300  |    25       | 0.829     | 0.684
Spanish  |  SVM   | stp-stemmed  |  pmi    |   1500  |    25       | 0.829     | 0.684

Catalan  |  SVM   | standard     |  None   |   NA    |    10       | 0.824     | 0.678



