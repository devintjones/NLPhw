import os
import sys
from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph

def main():
	try:
		sentences = sys.stdin.readlines()
		model_file = sys.argv[1]
	except:
		raise ValueError('''Usage: cat <file of sentences> | python parse.py <model_file>
		or, python parse.py <model_file>, type sentences and hit Ctrl+d''')
	
	if not os.path.isfile(model_file):
		raise ValueError('cant find the model file')
	
	# scrub list / remove line breaks
	sentences = [sent.rstrip() for sent in sentences]

	# generate dependency graph object from sentences
	depgraphs = [DependencyGraph.from_sentence(sent) for sent in sentences]

	# load model and parse
	tp = TransitionParser.load(model_file)
	parsed = tp.parse(depgraphs)
	
	# print to stdout. 
	# can cat this to a conll file for viewing with MaltEval
	for p in parsed:
		print(p.to_conll(10).encode('utf-8'))
	
	return

if __name__ == '__main__':
	main()
	pass
