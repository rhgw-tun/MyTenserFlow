# -*- coding: utf-8 -*-
"""
@author: rhg
"""

# Execution example

from MyAutoEncoder import ClassMyAutoEncoder

from sklearn import datasets
import numpy as np

bst = datasets.load_boston()
x_vals = np.array([i[0:13] for i in bst.data])

insMyAutoEncoder = ClassMyAutoEncoder()
W1 = insMyAutoEncoder.MyAutoEncoder(data = x_vals, layer = 6, epoch = 1000)