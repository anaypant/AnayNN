import numpy as np


def log_softmax(x, deriv=False):
    if deriv:
        return x * (1 - x)
    t = []
    for z in x:
        c = z.max()
        logsumexp = np.log(np.exp(z-c).sum())
        t.append((z-c-logsumexp).tolist())
    t = np.reshape(t, (len(t), len(t[0])))
    return t


def log_softmax_2(x):
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp


a = [np.array([0.1, 0.9])]
print(log_softmax(a[0], deriv=True))
print(log_softmax_2(a[0]))
