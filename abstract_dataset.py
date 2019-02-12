class AbstractDataset(object):
    def __init__(self):
        self.X = []
        self.y = []
        self.input_shape = 0
        self.n_classes = 0

    def load(self):
        assert False

    def save(self, filename):
        assert False
