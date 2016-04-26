import numpy
from multiprocessing import Process, Queue


def sequential(batch_size, num_samples, num_epochs):
    for epoch in range(num_epochs):
        indices = numpy.arange(num_samples)
        for i in range(0, num_samples - batch_size + 1, batch_size):
            slice_ = slice(i, i+batch_size)
            yield indices[slice_]


def random(batch_size, num_samples, num_epochs):
    for epoch in range(num_epochs):
        indices = numpy.random.permutation(num_samples)
        for i in range(0, num_samples - batch_size + 1, batch_size):
            slice_ = slice(i, i+batch_size)
            yield indices[slice_]


def batch_generator(data, index_generator, batch_function):
    for samples in index_generator:
        batch = data.get(samples)
        yield batch_function(batch)


class BatchFetcher(Process):
    def __init__(self, queue, batch_generator):
        self.queue = queue
        self.batch_generator

    def run(self):

        for batch in batch_generator():
            self.queue.put(batch)


class NumpyData(object):
    def __init__(self, array):
        self.array = array

    def get(self, indices):
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        return self.array[indices]