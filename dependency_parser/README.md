Title: NLP HW2
Author: Devin Jones
Uni: dj2374
Date: 2/28/2015

# 1
## a
Visualize dependency graph

dep_graph_visualizer.py

## b
A dependency graph is projective if none of the arcs in the graph cross

The algorithm to check whether a dependency graph is as follows:
For all parent/child arcs, ensure no nodes between child and parent connect to any nodes outside of child/parent
If there are arcs from in between parent/child to outside parent/child, the graph is non-projective.
Otherwise, the graph is projective.  

## c
Projective sentence:
* I went to the store
Non projective sentence:
* I ran to the store quickly which was Duane Reade. 

In above, 'ran' depends on 'quickly' and 'store' depends on 'Duane Reade'. These dependency arcs cross, making the syntax non-projective.


# 2
## a
Complete implementation of transition.py

## b
The performance of the parser using badfeatures.model is quite poor with an unlabeled attachement score (UAS) of 23% and labeled attachment score of 12%, compared to a target LAS of 70%. The badfeatures model exlcudes important information to the parser such as part of speech, lemma, and features for entries beyond the first in the stack and buffer. 

# 3
## a
### Feature 1
The feature that gave the greatest initial increase in LAS was the POS tag. By using the part of speech in the stack and buffer, the multinomial svm could better infer which shift to make to the stack and buffer. The 4 options are left arc, right arc, shuffle and reduce. This feature adds one additional computation, extracting the POS, to the complexity of building the feature sets. 

### Feature 2
From the literature from Dependency Parsing (Kubler, MacDonald, Nivre), I implemented identifying whether or not a verb occurs between the stack and buffer indexes of tokens. This marginally improved perfomance above the recommend features from chart 3.2. This feature increases the complexity by the max distance between the indexes of the stack and buffer because we have to check every cpostag between the stack and buffer. 

### Feature 3
From the same literature referenced above, I added features that depend on greater depths of the stack and buffer which improved perfomance of the model. These features were classified as included in most modern parsers. Referencing one object deeper in the stack and 3 deeper in the buffer significantly improved performance of the dependency parser. 

### Feature 4
If the deps tag is included in the model, more specifically the length of the dependent entries of the dependency tree, the performance across all languages explodes beyond 70% LAS even for Korean. The results after adding all mentioned features are below:

english model:
UAS: 0.862273965816
LAS: 0.812979935596
swedish model:
UAS: 0.839872535352
LAS: 0.710416251743
korean model:
UAS: 0.928157589803
LAS: 0.782155272306
danish model:
UAS: 0.870259481038
LAS: 0.771856287425


### Feature 3

## b 
Generate & save trained models for English, Dansih, Swedish, and Korean datasets.

## c
Score the above mentioned models. 
train_models.py

## d
The arc eager parser, as an alternative to the arc standard parser, is more memory efficient due to early composition. Memory efficiency is due to eager attachments being made with less bottom up evidence. Meanwhile, the arc standard parser must maintain larger stacks and has a larger memory footprint, but tends to be safer because attachments are made after an entire tree has been connected. 

The complexity of the arc-eager shift-reduce transition parser is linear in the number of words with one pass over each word to build the dependency graph. 

# 4
Create parse.py

