"""
Mark Fraser
m_fraser3@u.pacific.edu
COMP 151:  Project 4

A color-classification algorithm provided RGB values (0-255) and classification
"""


from numpy import (
    array,
    array2string,
    around,
    array_equal,
    count_nonzero,
    exp,
    float as np_float,
    fromstring,
    insert,
    ones,
    matrix,
    transpose,
    zeros
)

from numpy.random import (
    rand
)

from re import (
    split
)

from sys import (
    exit
)


class ColorClassifier(object):

    classifications = {"Red":   0,
                      "Blue":   1,
                      "Yellow": 2,
                      "Green":  3,
                      "Purple": 4,
                      "Orange": 5,
                      "Brown":  6,
                      "Pink":   7,
                      "Gray":   8
                      }
    rgb_range = 255 # range of possible RGB values (0-255)
    threshold = 0.5 # classification threshold

    def __init__(self, eta=None, epochs=None):
        super(ColorClassifier, self).__init__()
        self.w = rand(len(self.classifications),4) # initialize weights randomly
        self.eta = eta # learning rate
        self.epochs = epochs # number of iterations

    # loads and stores weights from provided weights file
    def load_weights(self, file):
        f = open(file, 'r')
        if f is None:
            exit('unable to open ' + file)
        lines = split(';', f.readline())
        for i, l in enumerate(lines):
            l = l.strip() # trim whitespace
            self.w[i] = array(split('\s+', l))

    # tests stored weights against data in provided data file + prints final
    # statistics to console
    def test(self, file):
        # statistics variables
        num_correct = 0 # number of correct classifications
        num_mulfire = 0 # number of multiple firings
        num_nofire = 0 # number of no fires
        n = len(self.classifications) # number of classification perceptrons
        neuron_stats = zeros((n, 3)) # per-neuron stats

        x, y = self._parse_data(file) # get data and classifications
        m, _ = x.shape # extract number of points
        w_t = transpose(self.w) # w^T pre-calculated to save on computations
        for i in range(m):
            r = self._forward_propagation(x[i])
            c = self._classify(r) # classify points to be 0 or 1
            o = self._get_correct_output(y[i]) # get target output

            # gather per-neuron stats
            for j in range(n):
                if c[j] == o[j]:
                    neuron_stats[j,0] += 1
                else:
                    neuron_stats[j,1] += 1 if c[j] == 1 and c[j] != o[j] else 0
                    neuron_stats[j,2] += 1 if c[j] == 0 and c[j] != o[j] else 0

            # gather aggregate statistics
            if array_equal(c,o):
                num_correct += 1
            else:
                nnz = count_nonzero(c) # number of non-zero entries
                if nnz > 1:
                    num_mulfire += 1
                elif nnz == 0:
                    num_nofire += 1

        # print results
        self._print_statistics(m, num_correct, num_mulfire, num_nofire,
                                                            neuron_stats)

    # trains stored weights on data points in provided file
    def train(self, file):
        x, y = self._parse_data(file) # get data and classifications
        m, _ = x.shape # extract number of points
        n, _ = self.w.shape # extract number of classifications

        # train data
        w_t = transpose(self.w) # w^T pre-calculated to save on computations
        for _ in range(self.epochs):
            for i in range(m):
                r = self._forward_propagation(x[i])
                c = self._classify(r) # classify points to be 0 or 1

                # perform training update
                o = self._get_correct_output(y[i]) # get target output
                for j in range(n):
                    self.w[j] = self.w[j] + (self.eta * (o[j]-c[j]) * x[i])

    # writes contents of stored weights to provided file
    def write(self, file):
        f = open(file, 'w')
        if f is None:
            exit('unable to open ' + file)
        
        n, _ = self.w.shape
        for i in range(n):
            s = array2string(self.w[i], separator=' ')
            # strip brackets from output and mark end of row
            s = s[1:len(s)-1] + ';'
            if i == n-1:
                s = s[:len(s)-1] # trim last semi-colon
            f.write(s)
        f.close()

    # classifies provided set of points (assumes sigmoid function has been
    # applied)
    def _classify(self, outputs):
        n = len(outputs)
        c = zeros(n)
        for i in range(n):
            if outputs[i] >= self.threshold:
                c[i] = 1
        return c

    # performs forward propagation and sigmoid on result
    def _forward_propagation(self, x):
        r = x.dot(transpose(self.w))
        # apply Sigmoid function
        for i in range(len(r)):
            r[i] = 1 / (1 +  exp(-r[i]))
        return r

    # returns tuple of point values and classifications obtained from data file
    def _parse_data(self, file):
        # prepare file contents
        f = open(file, 'r')
        if f is None:
            exit('unable to open ' + file)
        lines = f.readlines()

        # initialize points and classifications
        m = len(lines) # number of data points
        x = zeros((m,4)) # RGB + bias threshold
        y = zeros(m) # classifications

        # organize data
        for i in range(m):
            tmp = array(split(r'\s+', lines[i]))
            rgb = tmp[0:3].astype(np_float) / self.rgb_range # normalize data
            x[i,:] = insert(rgb, 0, 1)
            y[i] = self.classifications[tmp[3]]

        f.close()
        return (x, y)

    # create array of 0's except for correct answer
    def _get_correct_output(self, c):
        n, _ = self.w.shape
        y = zeros(n)
        y[int(c)] = 1
        return y

    # print provided statistics to console
    def _print_statistics(self, m, num_correct, num_mulfire, num_nofire,
                                                                neuron_stats):
        print('Aggregate Testing Results:')
        print('--------------------------')
        print('    Percentage Correct: '.ljust(32)
                                                + str(round(num_correct/m, 4)))
        print('    Percentage Multiple Firing: '.ljust(32)
                                                + str(round(num_mulfire/m, 4)))
        print('    Percentage No Firing: '.ljust(32)
                                                + str(round(num_nofire/m, 4)))
        print('\n')

        print('Per-Neuron Testing Results:')
        print('---------------------------')
        print('Correct   False Pos.  False Neg.'.rjust(43))
        for k, v in self.classifications.items():
            print((k + ':').rjust(10) + ' ' +
                '{}'.format(round(neuron_stats[v][0]/m,4)).ljust(10) +
                '{}'.format(round(neuron_stats[v][1]/m,4)).ljust(12) +
                '{}'.format(round(neuron_stats[v][2]/m,4)))
