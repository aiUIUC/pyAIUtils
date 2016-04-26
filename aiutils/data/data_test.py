import numpy as np
import data

def test_numpy_data_integration():
    temp = np.arange(100)
    array = np.tile(temp[:, None,  None, None], [1, 32, 32, 3])
    data_ = data.NumpyData(array)
    random_iterator = data.random(10, 100, 2)
    batcher = data.batch_generator(data_, random_iterator, lambda x: x)
    batch = batcher.next()
    assert batch.shape == (10, 32, 32, 3)