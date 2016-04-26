import numpy as np
import data
from multiprocessing import Queue


def test_numpy_data_integration():
    # Setup.
    batch_size = 10
    num_samples = 100
    num_epochs = 2
    temp = np.arange(num_samples)
    array = np.tile(temp[:, None,  None, None], [1, 32, 32, 3])

    # Create batcher.
    data_ = data.NumpyData(array)
    random_iterator = data.random(batch_size,
                                  num_samples,
                                  num_epochs)
    batcher = data.batch_generator(data_, random_iterator)

    # Get batch.
    batch = batcher.next()
    assert batch.shape == (batch_size, 32, 32, 3)


def test_batch_fetcher_integration():
    # Setup.
    batch_size = 10
    num_samples = 100
    num_epochs = 2
    temp = np.arange(num_samples)
    array = np.tile(temp[:, None,  None, None], [1, 32, 32, 3])

    # Create queue_batcher.
    data_ = data.NumpyData(array)
    random_iterator = data.random(batch_size,
                                  num_samples,
                                  num_epochs)
    batcher = data.batch_generator(data_, random_iterator)

    queue = Queue(maxsize=10)
    fetcher = data.BatchFetcher(queue, batcher)
    fetcher.start()

    queue_batcher = data.queue_generator(queue)

    # Get batches.
    for batch in queue_batcher:
        print batch.shape
    assert batch.shape == (batch_size, 32, 32, 3)


def test_make_queue_generator_integration():
    # Setup.
    batch_size = 10
    num_samples = 100
    num_epochs = 2
    temp = np.arange(num_samples)
    array = np.tile(temp[:, None,  None, None], [1, 32, 32, 3])

    # Create batcher.
    data_ = data.NumpyData(array)
    random_iterator = data.random(batch_size,
                                  num_samples,
                                  num_epochs)
    batcher = data.make_queue_generator(data_,
                                        random_iterator,
                                        queue_maxsize=10)

    # Get batches.
    for batch in batcher:
        print batch.shape
    assert batch.shape == (batch_size, 32, 32, 3)