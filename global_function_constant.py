#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.23
# Modified    :   2017.6.23
# Version     :   1.0

import re
# global constant
MAX_DEPTH = 5
VERSION = (1, 0, 1)
__version__ = '.'.join(map(str, VERSION))

# baisc_model_name_prefix
MODEL_NAME = 'model_store/basic_model'

# Traditional entropy.
ENTROPY1 = 'entropy1'

# Modified entropy that penalizes universally unique values.
ENTROPY2 = 'entropy2'

# Modified entropy that penalizes universally unique values
# as well as features with large numbers of values.
ENTROPY3 = 'entropy3'

DISCRETE_METRICS = [
    ENTROPY1,
    ENTROPY2,
    ENTROPY3,
]

# define the max value of a continous attribute
MAX_VALUE = 9999999999
# Simple statistical variance, the measure of how far a set of numbers
# is spread out.
VARIANCE1 = 'variance1'

# Like ENTROPY2, is the variance weighted to penalize attributes with
# universally unique values.
VARIANCE2 = 'variance2'

CONTINUOUS_METRICS = [
    VARIANCE1,
    VARIANCE2,
]

DEFAULT_DISCRETE_METRIC = ENTROPY1
DEFAULT_CONTINUOUS_METRIC = VARIANCE1

# Methods for aggregating the predictions of trees in a forest.
EQUAL_MEAN = 'equal-mean'
WEIGHTED_MEAN = 'weighted-mean'
BEST = 'best'
AGGREGATION_METHODS = [
    EQUAL_MEAN,
    WEIGHTED_MEAN,
    BEST,
]

# Forest growth algorithms.
GROW_RANDOM = 'random'
GROW_AUTO_MINI_BATCH = 'auto-mini-batch'
GROW_AUTO_INCREMENTAL = 'auto-incremental'
GROW_METHODS = [
    GROW_RANDOM,
    GROW_AUTO_MINI_BATCH,
    GROW_AUTO_INCREMENTAL,
]

# Data format names.
ATTR_TYPE_NOMINAL = NOM = 'nominal'
ATTR_TYPE_DISCRETE = DIS = 'discrete'
ATTR_TYPE_CONTINUOUS = CON = 'continuous'
ATTR_MODE_CLASS = CLS = 'class'
ATTR_HEADER_PATTERN = re.compile("([^,:]+):(nominal|discrete|continuous)(?::(class))?")

# -------global function
def get_mean(seq):
    """
    Batch mean calculation.
    """
    return sum(seq)/float(len(seq))

def get_variance(seq):
    """
    Batch variance calculation.
    """
    m = get_mean(seq)
    return sum((v-m)**2 for v in seq)/float(len(seq))

def standard_deviation(seq):
    return math.sqrt(get_variance(seq))

def mean_absolute_error(seq, correct):
    """
    Batch mean absolute error calculation.
    """
    assert len(seq) == len(correct)
    diffs = [abs(a-b) for a, b in zip(seq, correct)]
    return sum(diffs)/float(len(diffs))

def normalize(seq):
    """
    Scales each number in the sequence so that the sum of all numbers equals 1.
    """
    s = float(sum(seq))
    return [v/s for v in seq]

def erfcc(x):
    """
    Complementary error function.
    """
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * math.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r

def normcdf(x, mu, sigma):
    """
    Describes the probability that a real-valued random variable X with a given
    probability distribution will be found at a value less than or equal to X
    in a normal distribution.
    
    http://en.wikipedia.org/wiki/Cumulative_distribution_function
    """
    t = x-mu
    y = 0.5*erfcc(-t/(sigma*math.sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y

def normpdf(x, mu, sigma):
    """
    Describes the relative likelihood that a real-valued random variable X will
    take on a given value.
    
    http://en.wikipedia.org/wiki/Probability_density_function
    """
    u = (x-mu)/abs(sigma)
    y = (1/(math.sqrt(2*pi)*abs(sigma)))*math.exp(-u*u/2)
    return y

def normdist(x, mu, sigma, f=True):
    if f:
        y = normcdf(x, mu, sigma)
    else:
        y = normpdf(x, mu, sigma)
    return y

def normrange(x1, x2, mu, sigma, f=True):
    p1 = normdist(x1, mu, sigma, f)
    p2 = normdist(x2, mu, sigma, f)
    return abs(p1-p2)

def cmp(a, b): # pylint: disable=redefined-builtin
    return (a > b) - (a < b)


