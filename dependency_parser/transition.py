class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
		return -1

    	if conf.buffer[0] == 0:
		return -1

        idx_wi = conf.stack[len(conf.stack)-1]

	flag= True
	for (idx_parent,r,idx_child) in conf.arcs:
		if idx_child==idx_wi:
			flag=False
	if flag:
		conf.stack.pop()
		idx_wj = conf.buffer[0]

        	conf.arcs.append((idx_wj, relation, idx_wi))
	else:
		return -1


    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
	if not conf.stack:
		return -1
	idx_wi = conf.stack[len(conf.stack) - 1]
	flag = False
	for (idx_parent,r,idx_child) in conf.arcs:
		if idx_child == idx_wi:
			flag = True
	if flag:
		conf.stack.pop()
	else:
        	return -1

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
	if not conf.buffer:
		return -1
	idx_wi=conf.buffer.pop(0)
	conf.stack.append(idx_wi)

