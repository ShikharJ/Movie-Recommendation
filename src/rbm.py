from basic import *
import theano


class RBM(Basic):

	def __init__(self, epochs=100, input=None, visible=784, hidden=500, W=None, hidden_bias=None, visible_bias=None, numpy_rng=None, theano_rng=None):
		super(SingularValueDecomposition, self).__init__()
		self.epochs = epochs
		self.visible = visible
		self.hidden = hidden
		
		if numpy_rng is None:
			numpy_rng = numpy.random.RandomState(1234)
		
		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		if W is None:
			initial_W = numpy.asarray(numpy_rng.uniform(low = -4 * numpy.sqrt(6.0 / (hidden + visible)), high = 4 * numpy.sqrt(6.0 / (hidden + visible)), size=(visible, hidden)), dtype=theano.config.floatX)
			W = theano.shared(value=initial_W, name='W', borrow=True)

		if hidden_bias is None:
			hidden_bias =  theano.shared(value=numpy.zeros(hidden, dtype=theano.config.floatX), name='hidden_bias', borrow=True)

		if visible_bias is None:
			visible_bias =  theano.shared(value=numpy.zeros(visible, dtype=theano.config.floatX), name='visible_bias', borrow=True)

		self.input = input
		if not input:
			self.input = theano.tensor.matrix('input')

		self.W = W
		self.hidden_bias = hidden_bias
		self.visible_bias = visible_bias
		self.theano_rng = theano_rng
		self.params = [self.W, self.hidden_bias, self.visible_bias]

	def propup(self, visible):
		pre_sigmoid_activation = theano.tensor.dot(visible, self.W) + self.hidden_bias
		return [pre_sigmoid_activation, theano.tensor.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_hidden_given_visible(self, visible0):
		pre_sigmoid_hidden1, hidden1_mean = self.propup(visible0)
		hidden1 = self.theano_rng.binomial(size=hidden1_mean.shape, n=1, p=hidden1_mean, dtype=theano.config.floatX)
		return [pre_sigmoid_hidden1, hidden1_mean, hidden1]

	def propdown(self, hidden):
		pre_sigmoid_activation = theano.tensor.dot(hidden, self.W) + self.visible_bias
		return [pre_sigmoid_activation, theano.tensor.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_visible_given_hidden(self, hidden0):
		pre_sigmoid_visible1,  visible1_mean = self.propdown(hidden0)
		visible1 = self.theano_rng.binomial(size=visible1_mean.shape, n=1, p=visible1_mean, dtype=theano.config.floatX)
		return [pre_sigmoid_visible1, visible1_mean, visible1]

	def gibbs_hidden_sampling(self, hidden0):
		pre_sigmoid_visible1, visible1_mean, visible1 = self.sample_visible_given_hidden(hidden0)
		pre_sigmoid_hidden1, hidden1_mean, hidden1 = self.sample_hidden_given_visible(visible1)
		return [pre_sigmoid_visible1 visible1_mean, visible1, pre_sigmoid_hidden1, hidden1_mean, hidden1]

	def gibbs_visible_sampling(self, visible0):
		pre_sigmoid_hidden1, hidden1_mean, hidden1 = self.sample_hidden_given_visible(visible0)
		pre_sigmoid_visible1, visible1_mean, visible1 = self.sample_visible_given_hidden(hidden1)
		return [pre_sigmoid_hidden1, hidden1_mean, hidden1, pre_sigmoid_visible1 visible1_mean, visible1]

	def free_energy(self, visible_sample):
		WX_b = theano.tensor.dot(visible_sample, self.W) + self.hidden_bias
		visible_bias_term = theano.tensor.dot(visible_sample, self.visible_bias)
		hidden_term = theano.tensor.sum(theano.tensor.log(1 + theano.tensor.exp(WX_b)), axis=1)
		return -hidden_term - visible_bias_term

	def pseudo_likelihood_cost(self, updates):
		"""Stochastic Approximation To The Pseudo-Likelihood"""
		bit_i_index = theano.shared(value=0, name='bit_i_index')
		xi = theano.tensor.round(self.input)
		fe_xi = self.free_energy(xi)
		xi_flip = theano.tensor.set_subtensor(xi[:, bit_i_index], 1 - xi[:, bit_i_index])
		fe_xi_flip = self.free_energy(xi_flip)
		cost = theano.tensor.mean(self.visible, * theano.tensor.log(theano.tensor.nnet.sigmoid(fe_xi_flip - fe_xi)))
		updates[bit_i_index] = (bit_i_index + 1) % self.visible
		return cost

	def reconstruction_cost(self, updates, pre_sigmoid_nv):
		cross_entropy = theano.tensor.mean(theano.tensor.sum(self.input * theano.tensor.log(theano.tensor.nnet.sigmoid(pre_sigmoid_nv)) + (1 - self.input) * theano.tensor.log(1 - theano.tensor.nnet.sigmoid(pre_sigmoid_nv)), axis=1))
		return cross_entropy

	def cost_updates(self, learning_rate=0.1, persistent=None, k=1):
		#TODO
		pre_sigmoid_positive_hidden, positive_hidden_mean, positive_hidden_sample = self.sample_hidden_given_visible(self.input)
		if persistent is None:
			chain_start = positive_hidden_sample
		else:
			chain_start = persistent