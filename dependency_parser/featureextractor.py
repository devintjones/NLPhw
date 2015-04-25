from nltk.compat import python_2_unicode_compatible


@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most,right_most,left_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        """
        Think of some of your own features here! Some standard features are
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre

        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        """

        result = []

        if stack:
            stack_idx0 = stack[-1]
            token = tokens[stack_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)

            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                result.append('STK_0_LEMMA_' + token['lemma'])
            
            if FeatureExtractor._check_informative(token['tag']):
		result.append('STK_0_POS_' + token['tag'])
	

	    # Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most ,right_most,left_most= FeatureExtractor.find_left_right_dependencies(stack_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)
	    if buffer:
                stkmax = max(stack)
                buffer_idx0 = buffer[0]
		dist = buffer_idx0-stkmax
		result.append('STKBUFDIST_0_DIST_{0}'.format(dist))
		if dist > 1:
			for i in range(stkmax+1,buffer_idx0):
				if tokens[i].get('ctag') == 'VERB':
					result.append('STKBUFPOS_0_POS_{0}'.format('VERB'))
					break
            if len(stack)>1:
		stack_idx1 = stack[-2]
		token1 = tokens[stack_idx1]
		if FeatureExtractor._check_informative(token1['tag']):
                    result.append('STK_1_POS' + token1['tag'])
		    #if FeatureExtractor._check_informative(token['tag']):
                    #    result.append('STK_01_POS' + token['tag'] + token1['tag'])
            	if 'feats' in token1 and FeatureExtractor._check_informative(token1['feats']):
                    feats = token1['feats'].split("|")
                    for feat in feats:
                        result.append('STK_1_FEATS_' + feat)
		    
        	    
	if buffer:
            buffer_idx0 = buffer[0]
            token = tokens[buffer_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                result.append('BUF_0_LEMMA_' + token['lemma'])
            if FeatureExtractor._check_informative(token['tag']):
                result.append('BUF_0_POS_' + token['tag'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)

            dep_left_most, dep_right_most ,right_most,left_most= FeatureExtractor.find_left_right_dependencies(buffer_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)
	    if len(buffer)>1:
		buffer_idx1 = buffer[1]
		token1 = tokens[buffer_idx1]
                if FeatureExtractor._check_informative(token1['word'], True):
                    result.append('BUF_1_FORM_' + token1['word'])
                if FeatureExtractor._check_informative(token1['tag']):
                    result.append('BUF_1_POS_' + token['tag'])
	    
	    if len(buffer)>2:
		buffer_idx2 = buffer[2]
		token2 = tokens[buffer_idx2]
                if FeatureExtractor._check_informative(token2['tag']):
                    result.append('BUF_2_POS_' + token2['tag'])
                    if FeatureExtractor._check_informative(token['tag']):
                        result.append('BUF_02_POS_' + token1['tag'] + token2['tag'])
	    if len(buffer)>3:
		buffer_idx3 = buffer[3]
		token = tokens[buffer_idx3]
                if FeatureExtractor._check_informative(token['tag']):
                    result.append('BUF_3_POS_' + token['tag'])

        return result
