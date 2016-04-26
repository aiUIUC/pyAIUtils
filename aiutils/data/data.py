"""Utility functions of making batch generation easier."""
import numpy as np
from multiprocessing import Process, Queue


def sequential(batch_size, num_samples, num_epochs):
    """Generate sequence indices.
    """
    for epoch in range(num_epochs):
        indices = np.arange(num_samples)
        for i in range(0, num_samples - batch_size + 1, batch_size):
            slice_ = slice(i, i+batch_size)
            yield indices[slice_]


def random(batch_size, num_samples, num_epochs):
    """Generate random indices.
    """
    for epoch in range(num_epochs):
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples - batch_size + 1, batch_size):
            slice_ = slice(i, i+batch_size)
            yield indices[slice_]


def batch_generator(data, index_generator, batch_function):
    """Generate batches of data.
    """
    for samples in index_generator:
        batch = data.get(samples)
        yield batch_function(batch)


def queue_generator(queue, sentinel=None):
    """Create a generator from a queue.
    """
    while True:
        value = queue.get()
        if value is not sentinel:
            yield value
        else:
            return


def make_queue_generator(data, index_generator, batch_function, queue_maxsize):
    """Create an asynchronous batch generator.
    """
    batcher = batch_generator(data, index_generator, batch_function)

    queue = Queue(maxsize=queue_maxsize)
    fetcher = BatchFetcher(queue, batcher)
    fetcher.start()

    queue_batcher = queue_generator(queue)
    return queue_batcher


class BatchFetcher(Process):
    def __init__(self, queue, batch_generator):
        super(BatchFetcher, self).__init__()
        self.queue = queue
        self.batch_generator = batch_generator

    def run(self):

        for batch in self.batch_generator:
            self.queue.put(batch)

        # Signal that fetcher is done.
        self.queue.put(None)


class NumpyData(object):
    def __init__(self, array):
        self.array = array

    def get(self, indices):
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        return self.array[indices]