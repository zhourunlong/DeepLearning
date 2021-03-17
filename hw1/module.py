# You need to complete the code in TODO
import numpy as np

###########################################################
# DO NOT MODIFIED THE CODE BELOW

class Initializer:

    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    def init(self, shape):
        raise NotImplementedError

class XavierUniform(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feedforward neural networks"
    Glorot, X. & Bengio, Y. (2010)
    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))
    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = self.get_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low=-a, high=a, size=shape)

    def get_fans(self, shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out

class Constant(Initializer):

    def __init__(self, val):
        self._val = val

    def init(self, shape):
        return np.full(shape=shape, fill_value=self._val)


class Zeros(Constant):

    def __init__(self):
        super(Zeros, self).__init__(0.0)


class module():

    def __init__(self, **kwargs):
        self.params = {p: None for p in self.param_names}
        self.ut_params = {p: None for p in self.ut_param_names}

        self.grads = {}
        self.shapes = {}
        
        self.training = True
        self.is_init = False

    def _forward(self, X, **kwargs):
        raise NotImplementedError

    def _backward(self, d, **kwargs):
        raise NotImplementedError
    
    def set_phase(self, phase):
        self.training = phase.lower() == "train"

    @property
    def name(self):
        return self.__class__.__name__
    
    def __repr__(self):
        shape = None if not self.shapes else self.shapes
        return "module: %s \t shape: %s" % (self.name, shape)

    @property
    def param_names(self):
        return ()

    def _init_params(self):
        for p in self.param_names:
            self.params[p] = self.initializers[p](self.shapes[p])
        self.is_init = True
    
    @property
    def ut_param_names(self):
        return ()
#########################################################################

class Linear(module):
    def __init__(self, d_in, d_out, w_init=XavierUniform(), b_init=Zeros()):

        super().__init__()
        
        self.initializers = {
            "weight": w_init,
            "bias": b_init,
        }

        self.input = None
        self.d_in = d_in
        self.d_out = d_out

        if d_in:
            self.shapes = {
                "weight": [d_in, d_out],
                "bias": [d_out]
            }

            self._init_params()

    def _forward(self, inputs):
        if not self.is_init:
            d_in = inputs.shape[-1]
            self.shapes = {
                "weight": [d_in, self.d_out],
                "bias": [self.d_out]
            }
            self.d_in = d_in
            self._init_params()

        out = inputs @ self.params['weight'] + self.params['bias']
        self.input = inputs
        return out

    def _backward(self, grad):
        gw = np.zeros(self.params["weight"].shape)
        for i in range(self.d_in):
            for j in range(self.d_out):
                gw[i][j] = self.input[0][i] * grad[0][j]

        self.grads = {
            "weight": gw,
            "bias": grad
        }

        return grad @ self.params['weight'].T

    @property
    def param_names(self):
        return ('weight', 'bias')
    
    @property
    def weight(self):
        return self.params['weight']
    
    @property
    def bias(self):
        return self.params['bias']

class Reshape(module):

    def __init__(self, *output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.input_shape = None

    def _forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], *self.output_shape)

    def _backward(self, grad):
        return grad.reshape(*self.input_shape)

class Conv2D(module):
    """
    Implement 2D convolution layer
    :param kernel: A list/tuple of int that has length 4 (in_channels, height, width, out_channels)
    :param stride: A list/tuple of int that has length 2 (height, width)
    :param padding: String ["SAME", "VALID"]
    :param w_init: weight initializer
    :param b_init: bias initializer
    """
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {"weight": w_init, "bias": b_init}
        self.shapes = {"weight": self.kernel_shape, "bias": self.kernel_shape[-1]}

        self.padding_mode = padding
        self.padding = None

        self._init_params()

    def _forward(self, inputs):
        """
        :param inputs:  shape (batch_size, in_c, in_h, in_w)
        :return outputs: shape (batch_size, out_c, out_h, out_w)
        where batch size is the number of images
        """
        in_c, k_h, k_w, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(inputs)
        self.X = X
        bsz, _, h, w = X.shape

        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1
        Y = np.zeros([bsz, out_c, out_h, out_w])
        for in_c_i in range(in_c):
            for out_c_i in range(out_c):
                kernel = self.params['weight'][in_c_i, :, :, out_c_i]
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        patch = X[:, in_c_i, r_start: r_start+k_h, c_start: c_start+k_w] * kernel
                        Y[:, out_c_i, r, c] += patch.reshape(bsz, -1).sum(axis=-1)
        self.out_shape = (out_h, out_w)
        self.input = inputs
        return Y + self.params['bias'].reshape(1, -1, 1, 1)

    def _backward(self, grad):
        """
        Compute gradients w.r.t layer parameters and backward gradients.
        :param grad: gradients from previous layer 
            with shape (batch_size, out_c, out_h, out_w)
        :return d_in: gradients to next layers 
            with shape (batch_size, in_c, in_h, in_w)
        """
        out_h, out_w = self.out_shape
        s_h, s_w = self.stride
        in_c, k_h, k_w, out_c = self.kernel_shape
        rpad, cpad = self.padding[2: 4]
        batch_sz, _, in_h, in_w = self.input.shape

        ret = np.zeros(self.input.shape, dtype=np.float32)
        gw = np.zeros((batch_sz,) + self.params["weight"].shape)
        gb = np.zeros((batch_sz,) + self.params["bias"].shape)

        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                for batch in range(batch_sz):
                    for channel_out in range(out_c):
                        gb[batch][channel_out] += grad[batch][channel_out][r][c]
                    for channel in range(in_c):
                        for i in range(k_h):
                            rpos = r_start + i
                            for j in range(k_w):
                                cpos = c_start + j
                                for channel_out in range(out_c):
                                    gw[batch][channel][i][j][channel_out] += grad[batch][channel_out][r][c] * self.X[batch][channel][rpos][cpos]
                                if (rpos < rpad[0] or rpos >= in_h + rpad[0]):
                                    continue
                                if (cpos < cpad[0] or cpos >= in_w + cpad[0]):
                                    continue
                                tmp = 0
                                for channel_out in range(out_c):
                                    tmp += grad[batch][channel_out][r][c] * self.params['weight'][channel][i][j][channel_out]
                                ret[batch][channel][rpos - rpad[0]][cpos - cpad[0]] += tmp
        
        self.grads = {
            "weight": gw,
            "bias": gb
        }

        return ret

    def _inputs_preprocess(self, inputs):
        _, _, in_h, in_w = inputs.shape
        _, k_h, k_w, _ = self.kernel_shape
        # padding calculation
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)
        return np.pad(inputs, pad_width=self.padding, mode="constant")

    def get_padding_2d(self, in_shape, k_shape, mode):

        def get_padding_1d(w, k):
            if mode == "SAME":
                pads = (w - 1) + k - w
                half = pads // 2
                padding = (half, half) if pads % 2 == 0 else (half, half + 1)
            else:
                padding = (0, 0)
            return padding

        h_pad = get_padding_1d(in_shape[0], k_shape[0])
        w_pad = get_padding_1d(in_shape[1], k_shape[1])
        return (0, 0), (0, 0), h_pad, w_pad

    @property
    def param_names(self):
        return "weight", "bias"

    @property
    def weight(self):
        return self.params['weight']
    
    @property
    def bias(self):
        return self.params['bias']


class MaxPool2D(module):

    def __init__(self, pool_size, stride, padding="VALID"):
        """
        Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__()
        self.kernel_shape = pool_size
        self.stride = stride

        self.padding_mode = padding
        self.padding = None

        self.input = None
        self.X = None

    def _forward(self, inputs):
        """
        :param inputs:  shape (batch_size, in_c, in_h, in_w)
        """
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_c, in_h, in_w = inputs.shape

        # zero-padding
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)
        X = np.pad(inputs, pad_width=self.padding, mode="constant")

        padded_h, padded_w = X.shape[2:4]
    
        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        # construct output matrix and argmax matrix
        max_pool = np.empty(shape=(batch_sz, in_c, out_h, out_w))
        argmax = np.empty(shape=(batch_sz, in_c, out_h, out_w), dtype=int)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                pool = X[:, :, r_start: r_start+k_h, c_start: c_start+k_w]
                pool = pool.reshape((batch_sz, in_c, -1))

                _argmax = np.argmax(pool, axis=2)[:, :, np.newaxis]
                argmax[:, :, r, c] = _argmax.squeeze(axis=2)

                # get max elements
                _max_pool = np.take_along_axis(pool, _argmax, axis=2).squeeze(axis=2)
                max_pool[:, :, r, c] = _max_pool

        self.X_shape = X.shape
        self.in_shape = inputs.shape
        self.out_shape = (out_h, out_w)
        self.argmax = argmax
        return max_pool

    def _backward(self, grad):
        out_h, out_w = self.out_shape
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        rpad, cpad = self.padding[2: 4]
        batch_sz, in_c, in_h, in_w = self.X_shape
        
        ret = np.zeros(self.in_shape, dtype=np.float32)
        
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                rpos = r_start + self.argmax[:, :, r, c] // k_w
                cpos = c_start + self.argmax[:, :, r, c] % k_w
                for batch in range(batch_sz):
                    for channel in range(in_c):
                        rr = rpos[batch, channel]
                        cc = cpos[batch, channel]
                        if (rr >= rpad[0] and rr < in_h - rpad[1] and cc >= cpad[0] and cc < in_w - cpad[1]):
                            ret[batch][channel][rr - rpad[0]][cc - cpad[0]] += grad[batch][channel][r][c]
        
        return ret

    def get_padding_2d(self, in_shape, k_shape, mode):

        def get_padding_1d(w, k):
            if mode == "SAME":
                pads = (w - 1) + k - w
                half = pads // 2
                padding = (half, half) if pads % 2 == 0 else (half, half + 1)
            else:
                padding = (0, 0)
            return padding

        h_pad = get_padding_1d(in_shape[0], k_shape[0])
        w_pad = get_padding_1d(in_shape[1], k_shape[1])
        return (0, 0), (0, 0), h_pad, w_pad


class Activation(module):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def _forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def func(self, x):
        raise NotImplementedError


class ReLU(Activation):
    def __init__(self):
        self.output = None

    def func(self, x):
        self.output = np.maximum(x, 0.0)
        return self.output

    def _backward(self, grad):
        tmp = self.output
        tmp[tmp > 0] = 1
        return grad * tmp


class Tanh(Activation):
    def __init__(self):
        self.output = None

    def func(self, x):
        self.output = np.tanh(x)
        return self.output

    def _backward(self, grad):
        grad = grad * (1 - self.output * self.output)
        return grad


class Softmax(module):
    def __init__(self):
        self.input = None
        self.output = None

    def _forward(self, inputs, axis=-1):
        Y = np.exp(inputs - inputs.max())
        Z = Y / np.sum(Y, axis=axis, keepdims=True)
        self.input = inputs
        self.output = Z
        return Z # distribution

    def _backward(self, grad):
        X = self.input
        Z = self.output
        dX = np.zeros(X.shape, dtype=np.float32)
        N = Z.shape[0]
        for n in range(N):
            J = -np.outer(Z[n,  :], Z[n,  :]) + np.diag(Z[n, :])
            dX[n,:] = np.dot(J, grad[n,:])
        return dX


if __name__ == "__main__":

    layers = [
            Conv2D([2, 3, 3, 10]),
            ReLU(),
            MaxPool2D([3, 3], [2, 2]),
            Reshape(-1),
            Linear(10, 10),
            Tanh(),
        ]
    layers[0].params["weight"] = np.array(
        [[[[ 0.10034741, -0.10923018, -0.13951169,  0.02620922,
           0.11209463, -0.03927368,  0.2455522 ,  0.09440251,
          -0.00973911, -0.0551014 ],
         [-0.08009744,  0.11698803, -0.03137447, -0.22489624,
          -0.05207429,  0.12155709, -0.16216859, -0.16576429,
           0.01611499,  0.01625606],
         [ 0.06864581,  0.17847365,  0.11464144,  0.05670569,
           0.11361383, -0.09042443, -0.07059199, -0.13879062,
          -0.10536136,  0.06689657]],

        [[-0.208334  , -0.03386239, -0.03531212, -0.00322536,
          -0.03788247, -0.09588832, -0.03761636,  0.20092505,
           0.22685647,  0.00093809],
         [ 0.06330945, -0.19632441, -0.09332216, -0.04350284,
           0.18709384, -0.1274559 , -0.00866532,  0.24800156,
           0.0099521 ,  0.05766132],
         [-0.1937654 ,  0.16667984,  0.05263836,  0.02301866,
          -0.08030899, -0.10004608, -0.04238123,  0.09260008,
           0.19176605,  0.00532325]],

        [[ 0.08647767,  0.04389243,  0.06379496,  0.08922312,
           0.17485274, -0.2128848 ,  0.13467704, -0.1309234 ,
          -0.15617682,  0.03700767],
         [-0.20649141,  0.19680719,  0.06499291,  0.11411078,
          -0.2471389 ,  0.04823145,  0.02900326, -0.1741877 ,
          -0.1771956 ,  0.09986747],
         [-0.09256576,  0.09804958,  0.02777646, -0.05671893,
           0.21713808,  0.17450929, -0.07283475, -0.23311245,
          -0.09971547, -0.05200206]]],


       [[[ 0.10468353,  0.25300628, -0.07359204,  0.13409732,
           0.04759048,  0.09791245, -0.17818803, -0.05164933,
          -0.13235886, -0.07995545],
         [ 0.00670526,  0.08510415, -0.20128378, -0.18852185,
          -0.0909241 ,  0.08251962,  0.17697941,  0.0272014 ,
           0.18103799, -0.05881954],
         [-0.0935763 , -0.07443489, -0.16799624,  0.16809557,
          -0.08239949,  0.02674822,  0.04012047,  0.01099809,
          -0.25400403,  0.24942434]],

        [[ 0.2070298 , -0.14932613, -0.10598685,  0.01022026,
           0.20527782,  0.24701637, -0.12383634,  0.03287163,
           0.15678546, -0.05395091],
         [ 0.11802146, -0.17311034,  0.05143219,  0.1868667 ,
           0.24696055, -0.21484058, -0.03659691, -0.15090589,
          -0.02521261,  0.02439543],
         [-0.20770998, -0.10375416,  0.21839033,  0.03524392,
          -0.02175199,  0.12948939,  0.12353204, -0.23056503,
           0.10659301,  0.17326987]],

        [[-0.17062354,  0.1435208 , -0.10902726, -0.09884633,
           0.08440794, -0.19848298,  0.08420925,  0.19809937,
           0.10026675, -0.03047777],
         [-0.03155725,  0.13539886,  0.03352691, -0.21201183,
           0.04222458,  0.16080765, -0.08321898,  0.21838641,
           0.1280547 ,  0.03782839],
         [ 0.12852815, -0.21495132,  0.18355937,  0.16420949,
           0.20934355, -0.18967807, -0.21360746, -0.18468067,
          -0.05139272, -0.03866057]]]], dtype=np.float32)
    
    layers[4].params['weight'] = np.array(
        [[ 0.06815682, -0.41381145, -0.32710046,  0.34138927, -0.03506786,
         0.33732942, -0.5395874 ,  0.056517  ,  0.47315797,  0.0900187 ],
       [-0.321956  ,  0.23854145, -0.13256437,  0.18445538, -0.51560444,
         0.14887139, -0.51245147,  0.26814377, -0.02967232, -0.41434735],
       [ 0.04670532, -0.47457483,  0.1680028 ,  0.54343534,  0.29511   ,
         0.08081549, -0.43529126,  0.21890727,  0.17655055, -0.49393934],
       [ 0.32019785,  0.020503  , -0.08120787,  0.31569323, -0.09687106,
        -0.02078467, -0.34875813, -0.19573534,  0.37851244, -0.34297976],
       [-0.09060311,  0.53571045, -0.28854045,  0.45661694,  0.45833147,
        -0.44771242, -0.03981645,  0.00242787, -0.20411544, -0.4958647 ],
       [-0.2829692 , -0.4430751 , -0.28673285,  0.33716825,  0.43267703,
        -0.50037426, -0.21695638,  0.5264514 ,  0.04327536,  0.13836497],
       [-0.54164785, -0.01653088,  0.5349371 , -0.13672741, -0.44142258,
        -0.04172686,  0.507196  , -0.17326587,  0.32745343,  0.32736975],
       [-0.319598  , -0.06203758,  0.23617937, -0.09802067, -0.3384849 ,
         0.51211435,  0.16513875,  0.4003412 , -0.5200709 , -0.2553419 ],
       [ 0.00226878, -0.47383627,  0.54009086, -0.28869098, -0.13770601,
        -0.31328425, -0.4322124 , -0.29305372, -0.21842065,  0.14727412],
       [-0.23964529, -0.15086825, -0.5412125 , -0.14709733,  0.03712023,
        -0.3702431 ,  0.10673262, -0.22659011,  0.14465407, -0.5190256 ]],
      dtype=np.float32)

    x = np.array(
        [[[[-0.8492061 ,  1.0779045 , -0.08430817, -0.16879   ],
         [-0.21486154, -1.3411552 ,  0.64358956,  0.09206003],
         [-1.0160062 ,  0.75887114,  1.7270764 ,  0.12690294],
         [-0.10278344, -0.57746404, -0.16129336, -0.42453188]],

        [[-0.51087075,  1.566093  , -0.35502744, -1.0280341 ],
         [-0.44904083, -0.91717184, -0.2204062 , -1.7094339 ],
         [-0.20429765,  0.96035904,  1.1546314 , -0.55592376],
         [-3.0869954 , -0.13439347,  1.047694  , -0.16260263]]]],
      dtype=np.float32)

    for l in layers:
        x = l._forward(x)

    y = x

    e = np.array(
        [[ 0.32257673, -0.43416652,  1.0324798 , -0.19434273,  0.59407026,
        -0.19911239,  0.2908744 ,  0.27966267,  0.24996994, -0.97430784]],
      dtype=np.float32)

    for l in reversed(layers):
        e = l._backward(e)

    dx = e

    target = np.array(
        [[[[ 0.05311276,  0.0012092 , -0.01387816, -0.09388599],
         [-0.07132251,  0.00879665,  0.15649047, -0.09754436],
         [-0.01795623,  0.00153671, -0.08608256,  0.04023483],
         [ 0.01426923, -0.07778768, -0.03364746, -0.02747378]],

        [[-0.07534513, -0.10424455, -0.00875467,  0.04870174],
         [ 0.01257615, -0.1349132 , -0.05684724, -0.09846888],
         [-0.0293199 ,  0.01961213, -0.02899574,  0.17384738],
         [-0.03952359,  0.0342231 ,  0.14877939, -0.00817257]]]],
      dtype=np.float32)

    if dx.shape == target.shape and np.abs(target - dx).max() < 1e-4:
        print("success!")
    else:
        print("fail!")
