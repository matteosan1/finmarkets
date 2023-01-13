from tensorflow.keras.models import Model

class LRP:
    """
    LRP - utility class to perform Layerwise Relevance Propagation. Currently it works only on dense layers.

    Params:
    -------
    model: tensorflow.keras.models.Model
        The Keras trained model to work with.
    epsilon: float
        Normalization factor for LRP step 2 calculation.
    """
    def __init__(self, model, epsilon=1e-9):
        self.model = model
        self.names, self.activations, self.weights, self.biases = [], [], [], []
        for layer in model.layers:
            self.names.append(layer.name)
            self.activations.append(Model(inputs=model.input, outputs=layer.output))
            w = layer.get_weights()
            self.weights.append(w[0])
            self.biases.append(w[1])
        self.nlayers = len(self.names)
        self.epsilon = epsilon

    def summary(self):
        """
        summary - shows a model summary with numerical details on the weights.

        Params:
        -------
        None
        """
        print ("nlayers: ", self.nlayers)
        print ("+++++++++++")
        for l in range(self.nlayers):
            print ("Layer name: {}".format(self.names[l]))
            print ("Weights:\n", self.weights[l])
            print ("Bias:\n", self.biases[l])
            print ("------------")

    def rho(self, w, l, types):
        if types[l][0] == 'gamma':
            return w + np.maximum(0, w) * types[l][1]
        else:
            return w

    def incr(self, z, l, types):
        z += self.epsilon
        if types[l][0] == 'epsilon':
            return z + (z**2).mean()**0.5 * types[l][1]
        else:
            return z
            
    def lrp(self, X, types):
        """
        lrp - performes the actual relevance propagation and output the input relevance matrix.
        
        Params:
        -------
        X: numpy.array
            The specific input on which we want to compute the relevance.
        types: list(tuple)
            List of tuples with LRP rule types: (type, param). Possible types are: zero, epsilon and gamma; 
            parameters are the gamma or the epsilon values.
        """
        y = self.model.predict(X)
        print ("LRP for input:\n", X)
        A = [np.array([X])] + [None]*self.nlayers
        for l in range(self.nlayers):
            A[l+1] = np.maximum(0, A[l].dot(self.weights[l]) + self.biases[l])
            #A[l+1] = self.activations[l].predict(X)

        R = [None]*self.nlayers + [A[self.nlayers]*y]
        for l in range(self.nlayers-1, 0, -1):
            w = self.rho(self.weights[l], l, types)
            b = self.rho(self.biases[l], l, types)

            z = self.incr(A[l].dot(w) + b, l, types)
            s = R[l+1]/z
            c = s.dot(w.T)
            R[l] = A[l]*c

        w = self.weights[0]
        wm, wp = np.minimum(0, w), np.maximum(0, w)
        lb = A[0]*0-1
        hb = A[0]*0+1
        z = A[0].dot(w) - lb.dot(wp) - hb.dot(wm)
        s = R[1]/z
        c, cp, cm = s.dot(w.T), s.dot(wp.T), s.dot(wm.T)
        R[0] = A[0]*c - lb*cp - hb*cm

        print ("Relevance:\n", R[0])
        return R
