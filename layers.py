# This file is part of "MPS Yokohama Deep Learning Series Day 08/06/2016"
#
# "MPS Yokohama Deep Learning Series Day 08/06/2016"
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "MPS Yokohama Deep Learning Series Day 08/06/2016"
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
# (c) Junya Kaneko <jyuneko@hotmail.com>

import numpy as np


########################################################################################################################
# Activation functions and those derivatives
########################################################################################################################
def sigmoid(s):
    return 1/(1 + np.exp(-s))


def d_sigmoid(y):
    return y * (1 - y)


def tanh(s):
    return np.tanh(s)


def d_tanh(s):
    return 1 - np.power(np.tanh(s), 2)


########################################################################################################################
# Evaluation functions and those derivatives
########################################################################################################################
def se(t, y):
    return ((t - y).T @ (t - y)).flatten()[0] / 2.0


def d_se(t, y):
    return -(t - y)


########################################################################################################################
# Layers
########################################################################################################################
class BaseLayer:
    def __init__(self, n_output, n_prev_output, f, df):
        self._W = np.random.randn(n_output, n_prev_output)
        self._b = np.random.randn(n_output, 1)
        self._f = f
        self._df = df
        self._y = None
        self._delta = None

    @property
    def W(self):
        return self._W

    @property
    def y(self):
        return self._y

    @property
    def delta(self):
        return self._delta

    def propagate_forward(self, x):
        self._y = self._f(self._W @ x + self._b)
        return self._y

    def propagate_backward(self, next_delta, next_W):
        if next_W is not None:
            self._delta = next_W.T @ next_delta * self._df(self._y)
        else:
            self._delta = next_delta * self._df(self._y)
        return self._delta

    def update(self, prev_y, epsilon):
        Delta_W = self._delta @ prev_y.T
        self._W -= epsilon * Delta_W
        self._b -= epsilon * self._delta


class SigmoidLayer(BaseLayer):
    def __init__(self, n_output, n_prev_output):
        super().__init__(n_output, n_prev_output, sigmoid, d_sigmoid)


class TanhLayer(BaseLayer):
    def __init__(self, n_output, n_prev_output):
        super().__init__(n_output, n_prev_output, tanh, d_tanh)
