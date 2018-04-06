from basic import *
import copy

N_IT = 1
ETA = 0.001

class RestrictedBoltzmannMachine(Basic):

	def __init__(self, epochs=100, f=60, learning_rate=0.007, k_u=0.05, k_m=0.05, k_b=0.05, bias=False):
		self.f = f
        self.K = 5
        self.h = np.random.rand(self.F) - 0.5
        self.featureBias = np.random.rand(self.F) - 0.5
        self.movieBias = np.random.rand(self.m, self.K) - 0.5
        self.w = np.random.rand(self.F, self.m, self.K) - 0.5

    def train(self):
        for it in range(N_IT):
            for u in users:
                data = copy.deepcopy(self.data[u])
                w = self.getW(user_movies[u])
                posAssociations, self.h = self.fwdProp(data, user_movies[u])
                visibleProb = self.bwdProp(self.h, user_movies[u])
                negAssociations, temp = self.fwdProp(visibleProb, user_movies[u])
                w += ETA * (posAssociations - negAssociations) / len(user_movies[u]) #might change len
                self.setW(user_movies[u], w)
                error = np.sum((data - visibleProb) ** 2)
                error = np.sqrt(error/len(data))
                print it, u, error

    def getW(self, movies):
        a = np.zeros((self.F, 1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.w[:,m,:], axis=1)), axis=1)
        return a[:,1:,]

    def setW(self, movies, w):
        it = 0
        for m in movies:
            self.w[:, m, :] += w[:, it, :]
            it += 1

    def getMovieBias(self, movies):
        a = np.zeros((1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.movieBias[m,:], axis=0)), axis=0)
        return a[1:,]

    def fwdProp(self, inp, movies):
        hiddenUnit = np.copy(self.featureBias)
        for j in range(self.F):
            hiddenUnit[j] += np.tensordot(inp, self.getW(movies)[j])
        hiddenProb = sigmoid(hiddenUnit)
        hiddenStates = hiddenProb > np.random.rand(self.F)
        hiddenAssociations = np.zeros((self.F, len(movies), self.K))    # Same as self.w for a single user case
        for j in range(self.F):
            hiddenAssociations[j] = hiddenProb[j] * inp
        return hiddenAssociations, hiddenStates

    def bwdProp(self, inp, movies):
        visibleUnit = self.getMovieBias(movies)
        for j in range(self.F):
            visibleUnit += inp[j] * self.getW(movies)[j]
        visibleProb = sigmoid(visibleUnit)
        return visibleProb

    def predict(self, u, i):
        w = self.getW(user_movies[u])
        #making predictions part Vq not given
        data = copy.deepcopy(self.data[u])
        probs = np.ones(5)
        mx, index = -1, 0
        for i in range(5):
            calc = 1.0
            for j in range(self.F):
                temp = np.tensordot(data, self.getW(user_movies[u])[j]) + self.featureBias[j]
                temp = 1.0 + np.exp(temp)
                calc *= temp
            probs[i] = calc
            if mx < probs[i]:
                index = i
                mx = probs[i]
        return index
