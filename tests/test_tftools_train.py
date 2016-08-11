import pytest
import tensorflow as tf

from context import train

def test_simple_multi_optimizer_momentum():
    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)

        loss = tf.abs(a - b)

        sess = tf.InteractiveSession()
        optimizer = train.MultiRateOptimizer(tf.train.MomentumOptimizer)
        optimizer.add_variables(
            [a], learning_rate=.1, other_params={'momentum': .01})
        optimizer.add_variables(
            [b], learning_rate=.05, other_params={'momentum': .01})

        opt = optimizer.minimize(loss)
        tf.initialize_all_variables().run()

        for i in range(1000):
            opt.run()

        float_eps = .001
        assert a.eval() - b.eval() < float_eps

        assert a.eval() - 1.75 < float_eps


def test_simple_multi_optimizer0():
    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)

        loss = tf.abs(a - b)

        sess = tf.InteractiveSession()
        optimizer = train.MultiRateOptimizer(tf.train.GradientDescentOptimizer)
        optimizer.add_variables([a], learning_rate=.1)
        optimizer.add_variables([b], learning_rate=.05)

        opt = optimizer.minimize(loss)
        tf.initialize_all_variables().run()

        for i in range(100):
            opt.run()

        float_eps = .001
        assert a.eval() - b.eval() < float_eps

        assert a.eval() - 1.75 < float_eps


def test_simple_multi_optimizer1():
    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)

        loss = tf.abs(a - b)

        sess = tf.InteractiveSession()
        optimizer = train.MultiRateOptimizer(tf.train.GradientDescentOptimizer)
        opt = optimizer.add_variables(
            [a], learning_rate=.1).add_variables(
                [b], learning_rate=.05).minimize(loss)

        tf.initialize_all_variables().run()

        for i in range(100):
            opt.run()

        float_eps = .001
        assert a.eval() - b.eval() < float_eps

        assert a.eval() - 1.75 < float_eps


def test_simple_multi_optimizer2():
    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)

        loss = tf.abs(a - b)

        sess = tf.InteractiveSession()
        optimizer = train.MultiRateOptimizer()
        optimizer.add_variables([a], tf.train.GradientDescentOptimizer(.1))
        optimizer.add_variables([b], tf.train.GradientDescentOptimizer(.05))

        opt = optimizer.minimize(loss)
        tf.initialize_all_variables().run()

        for i in range(100):
            opt.run()

        float_eps = .001
        assert a.eval() - b.eval() < float_eps

        assert a.eval() - 1.75 < float_eps


def test_multi_optimizer():
    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)
        c = tf.Variable(3.0)

        loss = (a * b - c)**2

        sess = tf.InteractiveSession()
        optimizer = train.MultiRateOptimizer()
        optimizer.add_variables([a, b], tf.train.GradientDescentOptimizer(.01))
        optimizer.add_variables([c], tf.train.GradientDescentOptimizer(.001))

        opt = optimizer.minimize(loss)
        tf.initialize_all_variables().run()
        opt.run()
        float_eps = .1
        assert abs(c.eval() - (3 - .002)) < float_eps * .002
        assert abs(b.eval() - (2 + .02)) < float_eps * .02
        assert abs(a.eval() - (1 + .04)) < float_eps * .04

        for i in range(100):
            opt.run()

        assert c.eval() - (a * b).eval() < .01
        assert c.eval() > 2.5
        assert (a * b).eval() > 2.5


def test_variable_reuse():
    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)

        loss = tf.abs(a - b)

        sess = tf.InteractiveSession()
        optimizer = train.MultiRateOptimizer()
        optimizer.add_variables([a, b], tf.train.GradientDescentOptimizer(.1))

        with pytest.raises(ValueError):
            optimizer.add_variables([b],
                                    tf.train.GradientDescentOptimizer(.05))

def test_no_default():
    g = tf.Graph()
    with g.as_default():
        a = tf.Variable(1.0)
        b = tf.Variable(2.0)

        loss = tf.abs(a - b)

        sess = tf.InteractiveSession()
        optimizer = train.MultiRateOptimizer()
        optimizer.add_variables([a], tf.train.GradientDescentOptimizer(.1))

        with pytest.raises(ValueError):
            optimizer.add_variables([b], learning_rate=.01)


