from basic import *
import theano
import os


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
		pre_sigmoid_positive_hidden, positive_hidden_mean, positive_hidden_sample = self.sample_hidden_given_visible(self.input)
		if persistent is None:
			chain_start = positive_hidden_sample
		else:
			chain_start = persistent
		(
			[pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates
		) = theano.scan(self.gibbs_hidden_sampling, output_info=[None, None, None, None, None, chain_start], n_steps=k, name="gibbs_hidden_sampling")

		chain_end = nv_samples[-1]
		cost = theano.tensor.mean(self.free_energy(self.input)) - theano.tensor.mean(self.free_energy(chain_end))
		gparams = theano.tensor.grad(cost, self.params, consider_constant=[chain_end])
		for gparam, param in zip(gparams, self.params):
			updates[param] = param - gparam * theano.tensor.cast(learning_rate, dtype=theano.config.floatX)
		if persistent:
			updates[persistent] = nh_samples[-1]
			monitoring_cost = self.pseudo_likelihood_cost(updates)
		else:
			monitoring_cost = self.reconstruction_cost(updates, pre_sigmoid_nvs[-1])
		return monitoring_cost, updates


def test_rbm(learning_rate=0.1, training_epochs=15, dataset=None, batch_size=20, chains=20, samples=10, output_folder=None, hidden=500):
	train_set_x, train_set_y = dataset[0]
	test_set_x, test_set_y = dataset[2]

	n_train_batches = train_set_x.shape[0] // batch_size

	index = theano.tensor.lscalar()
	x = theano.tensor.matrix('x')

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	persistent_chain = theano.shared(numpy.zeros((batch_size, hidden), dtype=theano.config.floatX), borrow=True)

	rbm = RBM(input=x, visible=28 * 28, hidden=hidden, numpy_rng=rng, theano_rng=theano_rng)

	cost, updates = rbm.cost_updates(learning_rate=learning_rate, persistent=persistent_chain, k=15)

	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)
	os.chdir(output_folder)

	train_rbm = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size]
		},
		name='train_rbm'
	)

	plotting_time = 0.
	start_time = timeit.default_timer()

	for epoch in range(training_epochs):
		mean_cost = []
		for batch_index in range(n_train_batches):
			mean_cost += [train_rbm(batch_index)]
		print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))
		plotting_start = timeit.default_timer()
		image = Image.fromarray(
			tile_raster_images(
				X=rbm.W.get_value(borrow=True).T,
				img_shape=(28, 28),
				tile_shape=(10, 10),
				tile_spacing=(1, 1)
			)
		)
		image.save('filters_at_epoch_%i.png' % epoch)
		plotting_stop = timeit.default_timer()
		plotting_time += (plotting_stop - plotting_start)

	end_time = timeit.default_timer()
	pretraining_time = (end_time - start_time) - plotting_time
	print ('Training took %f minutes' % (pretraining_time / 60.))
	number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
	test_idx = rng.randint(number_of_test_samples - n_chains)
	persistent_vis_chain = theano.shared(numpy.asarray(test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains], dtype=theano.config.floatX))
	plot_every = 1000
	(
		[
			presig_hids,
			hid_mfs,
			hid_samples,
			presig_vis,
			vis_mfs,
			vis_samples
		],
		updates
	) = theano.scan(
		rbm.gibbs_visible_sampling,
		outputs_info=[None, None, None, None, None, persistent_vis_chain],
		n_steps=plot_every,
		name="gibbs_visible_sampling"
	)
	updates.update({persistent_vis_chain: vis_samples[-1]})
	sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]], updates=updates, name='sample_fn')
	image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')
	for idx in range(n_samples):
		vis_mf, vis_sample = sample_fn()
		print(' ... plotting sample %d' % idx)
		image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(X=vis_mf, img_shape=(28, 28), tile_shape=(1, n_chains), tile_spacing=(1, 1))
	image = Image.fromarray(image_data)
	image.save('samples.png')
	os.chdir('../')
