import numpy as np
import torch
import time

# from sklearn.metrics import confusion_matrix

def nan2num(x, nan2=0, inf2=0, neginf2=0):
    x[x == float("Inf")] = inf2
    x[x == float("-Inf")] = neginf2
    x[x != x] = nan2
    return x

def to_tensor(x, dtype=None):
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if type(x) is np.ndarray:
        return torch.from_numpy(x).type(dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype)


def set_dtype(dtype=None):
    if dtype == 'float32' or dtype is None:
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError('Unsupported data type: %s; should be either float32 or float64' % dtype)


def window_data(X, window_length):
    """Window the sample, X, to a length of window_length centered at the middle of the original sample
    """
    return X[int(len(X) / 2 - window_length / 2):int(len(X) / 2 + window_length / 2)]


def accuracy_onehot(y_pred, y_label):
    """Compute the accuracy for a onehot
    """
    return (y_pred.argmax(dim=1) == y_label).float().mean().item()


def normalize_power(X):
    return X / torch.sum(X, dim=1, keepdim=True)


def tic():
    t = time.clock()
    return t


def toc(t0, t1):
    t = tic()
    if (t - t1) > 0.01:
        print('Elapsed time \x1b[1;32m%.2f s\x1b[0m,' % (t - t0),
              '(dt = \x1b[1;34m%.2f s\x1b[0m)' % (t - t1))
    else:
        print('Elapsed time \x1b[1;32m%.2f s\x1b[0m,' % (t - t0),
              '(dt = \x1b[1;34m%.3f ms\x1b[0m)' % ((t - t1)*1000))
    return t


def stat_cuda(msg=''):
    print('GPU memory usage ' + msg + ':')
    print('allocated: %dM (max %dM), cached: %dM (max %dM)'
          % (torch.cuda.memory_allocated() / 1024 / 1024,
             torch.cuda.max_memory_allocated() / 1024 / 1024,
             torch.cuda.memory_reserved() / 1024 / 1024,
             torch.cuda.max_memory_reserved() / 1024 / 1024))
