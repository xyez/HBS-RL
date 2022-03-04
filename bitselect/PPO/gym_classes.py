import numpy as np


class Space(object):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """

    def __init__(self, shape=None, dtype=None):
        import numpy as np  # takes about 300-400ms to import, so we load lazily
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        raise NotImplementedError

    def seed(self, seed):
        """Set the seed for this space's pseudo-random number generator. """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n


class Box(Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.

    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """

    def __init__(self, low=None, high=None, shape=None, dtype=None):
        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = low + np.zeros(shape)
            high = high + np.zeros(shape)
        if dtype is None:  # Autodetect type
            if (high == 255).all():
                dtype = np.uint8
            else:
                dtype = np.float32
            print("gym.spaces.Box autodetected dtype as {}. Please provide explicit dtype.".format(dtype))
        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        super(Box, self).__init__(shape, dtype)
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        high = self.high if self.dtype.kind == 'f' else self.high.astype('int64') + 1
        return self.np_random.uniform(low=self.low, high=high, size=self.low.shape).astype(self.dtype)

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


class Discrete(Space):
    """
    {0,1,...,n-1}

    Example usage:
    self.observation_space = spaces.Discrete(2)
    """

    def __init__(self, n):
        self.n = n
        super(Discrete, self).__init__((), np.int64)
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
