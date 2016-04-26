import numpy as np
import data
from multiprocessing import Queue


def test_numpy_data_integration():
    temp = np.arange(100)
    array = np.tile(temp[:, None,  None, None], [1, 32, 32, 3])
    data_ = data.NumpyData(array)
    random_iterator = data.random(10, 100, 2)
    batcher = data.batch_generator(data_, random_iterator, lambda x: x)
    batch = batcher.next()
    assert batch.shape == (10, 32, 32, 3)


def test_batch_fetcher_integration():
    temp = np.arange(100)
    array = np.tile(temp[:, None,  None, None], [1, 32, 32, 3])
    data_ = data.NumpyData(array)
    random_iterator = data.random(10, 100, 2)
    batcher = data.batch_generator(data_, random_iterator, lambda x: x)

    queue = Queue(maxsize=10)
    fetcher = data.BatchFetcher(queue, batcher)
    fetcher.start()

    queue_batcher = data.queue_generator(queue)
    # batch = queue_batcher.next()

    for batch in queue_batcher:
        print batch.shape
    assert batch.shape == (10, 32, 32, 3)


def test_make_queue_generator_integration():
    temp = np.arange(100)
    array = np.tile(temp[:, None,  None, None], [1, 32, 32, 3])
    data_ = data.NumpyData(array)
    random_iterator = data.random(10, 100, 2)
    batcher = data.make_queue_generator(data_,
        random_iterator,
        lambda x: x,
        queue_maxsize=10)

    for batch in batcher:
        print batch.shape
    assert batch.shape == (10, 32, 32, 3)