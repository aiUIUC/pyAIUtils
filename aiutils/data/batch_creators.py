"""Utility functions for making batch generation easier."""
import numpy as np
from multiprocessing import Process, Queue
from builtins import range
import itertools
from math import ceil


def sequential(batch_size, num_samples, num_epochs=1, offset=0):
    """Generate sequence indices in range [offset, offset+num_samples].
    
    Inputs:
        batch_size (int): number of samples to generate in each iteration
        num_samples (int): total number of data samples
        num_epochs (int): number of epochs or number of repetitions 
        offset (int): offset or index to start samples from

    Output: 
        Yields a generator with sequential sampling
    """
    for epoch in range(num_epochs) if num_epochs > 0 else itertools.count():
        indices = np.arange(num_samples) + offset
        for i in range(0, num_samples - batch_size + 1, batch_size):
            yield indices[i:i + batch_size]


def random(batch_size, num_samples, num_epochs=1, offset=0):
    """Generate random indices in range [offset, offset+num_samples].

    Inputs:
        batch_size (int): number of samples to generate in each iteration
        num_samples (int): total number of data samples
        num_epochs (int): number of epochs or number of repetitions 
        offset (int): offset or index to start samples from

    Output: 
        Yields a generator with random sampling with replacement
    """
    for epoch in range(num_epochs) if num_epochs > 0 else itertools.count():
        if num_samples < batch_size:
            indices = np.random.permutation(list(range(num_samples))*int(ceil(float(batch_size)/num_samples))) + offset
            indices = indices.tolist()
        else:
            indices = np.random.permutation(num_samples) + offset
            indices = indices.tolist()

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            yield indices[i:i + batch_size]


def batch_generator(data, index_generator, batch_function=None):
    """Generate batches of data.

    Input: 
        data (class): data class with get_data method which produces a batch 
            given a list of sample indices
        index_generator (generator): either a sequential or a random index 
            generator
        batch_function (function): a function that may be applied to each batch

    Output: 
        Yields a batch generator
    """
    for samples in index_generator:
        batch = data.get_data(samples)
        if batch_function:
            output = batch_function(batch)
        else:
            output = batch
        yield output


def async_batch_generator(data,
                          index_generator,
                          queue_maxsize,
                          batch_function=None):
    """Create an asynchronous batch generator that runs in a separate process.

    Input: 
        data (class): data class with get_data method which produces a batch 
            given a list of sample indices
        index_generator (generator): either a sequential or a random index 
            generator
        queue_maxsize (int): queue size for storing batches
        batch_function (function): a function that may be applied to each batch

    Output:
        Yields a batch generator that runs as a separate process
    """
    batcher = batch_generator(data, index_generator, batch_function)

    queue = Queue(maxsize=queue_maxsize)

    # Start the process to enqueue batches into the queue
    enqueuer = BatchEnqueuer(queue, batcher)
    enqueuer.start()

    queue_batcher = queue_generator(queue)
    return queue_batcher


class BatchEnqueuer(Process):
    def __init__(self, queue, batch_generator):
        super(BatchEnqueuer, self).__init__()
        self.queue = queue
        self.batch_generator = batch_generator

    def run(self):
        for batch in self.batch_generator:
            self.queue.put(batch)

        # Signal that fetcher is done.
        self.queue.put(None)


def queue_generator(queue, sentinel=None):
    """Create a generator from a multiprocessing queue.

    Input:
        queue (multiprocessing.Queue): The queue to be converted into generator
        sentinel: Value to be used as a sentinel in the queue. Generator 
            terminates when it hits the sentinel in the queue.
    """
    while True:
        value = queue.get()
        if value is not sentinel:
            yield value
        else:
            return


class NumpyData(object):
    """A simple data class for numpy array data.
    """

    def __init__(self, array):
        """
            Input -
            array (np.array): an array whose 0th dimension corresponds to 
                different data samples.
        """
        self.array = array

    def get_data(self, indices):
        """
        Input -
            indices (list of int): A list of indices into array

        Ouput -
            numpy array with size of 0th dimension same as len(indices). Can be
            thought of as a batch sampled from self.array
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        return self.array[indices]
