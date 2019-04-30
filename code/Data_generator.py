from keras.utils import Sequence
import glob, os, copy
import numpy as np
import skimage.measure


class data_generator(Sequence):
    def __init__(self,atlas_volume, path, batch_size=1, lower_threshold=-75, upper_threshold=100):
        self.atlas = atlas_volume
        self.path = path
        self.batch_size = batch_size
        self.lower = lower_threshold
        self.upper = upper_threshold
        self.get_training_data()
        self.shuffle()

    def get_training_data(self):
        self.train_vol_names = glob.glob(os.path.join(self.path, '*Registered_Data.npy'))

    def shuffle(self):
        self.load_file_list = copy.deepcopy(self.train_vol_names)

        perm = np.arange(len(self.train_vol_names))
        np.random.shuffle(perm)
        self.load_file_list = list(np.asarray(self.load_file_list)[perm])

    def __getitem__(self, item):
        z_max = self.atlas.shape[1]
        X = np.load(self.load_file_list[item])
        while X.shape[2] > self.atlas.shape[2]:
            X = skimage.measure.block_reduce(X, (1, 2, 2, 2, 1), np.average)
        if X.shape[1] > int(z_max):
            X = X[:,-z_max:,...]
        holder = self.atlas.shape - np.asarray(X.shape)
        val_differences = [[i,0] for i in holder]
        if np.max(val_differences) > 0:
            X = np.pad(X, val_differences, 'constant', constant_values=(-1000))
        X = self.normalize(X)
        return X

    def normalize(self, X):
        X[X<self.lower] = self.lower
        X[X > self.upper] = self.upper
        X = (X - self.lower)/(self.upper - self.lower)
        return X


if __name__ == '__main__':
    xxx = 1