import numpy as np
import data


def test_batch_generator_integration():
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
    for batch in batcher:
        assert batch.shape == (batch_size, 32, 32, 3)


def test_async_batch_generator_integration():
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
    batcher = data.async_batch_generator(data_,
                                         random_iterator,
                                         queue_maxsize=10)

    # Get batches.
    for batch in batcher:
        assert batch.shape == (batch_size, 32, 32, 3)