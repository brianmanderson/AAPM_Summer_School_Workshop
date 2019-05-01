from keras.utils import Sequence
import glob, os, copy, sys
import numpy as np
import skimage.measure


class data_generator(Sequence):
    def __init__(self,atlas_volume, path, batch_size=1, lower_threshold=-75, upper_threshold=100):
        self.atlas = atlas_volume
        self.volume_shape = atlas_volume.shape[1:-1]
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
        zeros = np.zeros((self.batch_size, *self.volume_shape, len(self.volume_shape)))
        return ([X, self.atlas], [self.atlas, zeros])

    def on_epoch_end(self):
        self.shuffle()

    def __len__(self):
        return len(self.load_file_list)


    def normalize(self, X):
        X[X<self.lower] = self.lower
        X[X > self.upper] = self.upper
        X = (X - self.lower)/(self.upper - self.lower)
        return X


def cvpr2018_gen(gen, atlas_vol_bs, batch_size=1):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def cvpr2018_gen_s2s(gen, batch_size=1):
    """ generator used for cvpr 2018 model for subject 2 subject registration """
    zeros = None
    while True:
        X1 = next(gen)[0]
        X2 = next(gen)[0]

        if zeros is None:
            volshape = X1.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        yield ([X1, X2], [X2, zeros])


def miccai2018_gen(gen, atlas_vol_bs, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if bidir:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, X, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def miccai2018_gen_s2s(gen, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    zeros = None
    while True:
        X = next(gen)[0]
        Y = next(gen)[0]
        if zeros is None:
            volshape = X.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        if bidir:
            yield ([X, Y], [Y, X, zeros])
        else:
            yield ([X, Y], [Y, zeros])


def example_gen(vol_names, batch_size=1, return_segs=False):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx])
            # X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # also return segmentations
        if return_segs:
            X_data = []
            for idx in idxes:
                X_seg = load_volfile(vol_names[idx].replace('norm', 'aseg'))
                # X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)

            if batch_size > 1:
                return_vals.append(np.concatenate(X_data, 0))
            else:
                return_vals.append(X_data[0])

        yield tuple(return_vals)


def load_example_by_name(vol_name, seg_name):
    """
    load a specific volume and segmentation
    """
    X = load_volfile(vol_name)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_seg = load_volfile(seg_name)
    X_seg = X_seg[np.newaxis, ..., np.newaxis]

    return_vals.append(X_seg)

    return tuple(return_vals)


def load_volfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data'
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz', '.npy')), 'Unknown data file'
    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
    elif datafile.endswith(('.npy')):
        z_max = 128
        # factor = 64
        X = np.load(datafile)
        if X.shape[1] > z_max:
            X = X[:, -z_max:, ...]
        X = skimage.measure.block_reduce(X, (1, 2, 2, 2, 1), np.average)
        holder = (1, int(z_max / 2), 256, 256, 1) - np.asarray(X.shape)
        val_differences = [[i, 0] for i in holder]
        X = np.pad(X, val_differences, 'constant', constant_values=(-1000))
        # X = X[:,:,factor:-factor,factor:-factor,:]
        #
    else:  # npz
        X = np.load(datafile)['vol_data']

    return X


if __name__ == '__main__':
    xxx = 1