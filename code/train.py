"""
train atlas-based alignment with CVPR2018 version of VoxelMorph 
"""

# python imports
import os
import glob
import sys
from Utils import plot_scroll_Image
from Data_generator import data_generator
# third-party imports
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model
import keras.backend as K
# project imports
import Network_Building
import losses
import skimage.measure
from tensorflow.python.client import device_lib

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen




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

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def normalize(X, lower, upper):
    X[X<lower] = lower
    X[X > upper] = upper
    X = (X - lower)/(upper - lower)
    return X

def train_model(data_dir,
          atlas_file, 
          layers,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          reg_param,
          steps_per_epoch,
          batch_size,
          load_model_file,
          data_loss,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc
    """

    # load atlas from provided files. The atlas we used is 160x192x224.
    def cvpr2018_gen(gen, atlas_vol_bs, batch_size=1):
        """ generator used for cvpr 2018 model """

        volshape = atlas_vol_bs.shape[1:-1]
        zeros = np.zeros((batch_size, *volshape, len(volshape)))
        while True:
            X = next(gen)[0]
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])
    atlas_vol = np.load(atlas_file) #['vol'][np.newaxis, ..., np.newaxis]
    # plot_scroll_Image(np.transpose(np.squeeze(atlas_vol), [1, 2, 0]))
    z_max = 64
    # atlas_vol = atlas_vol[:,:,factor:-factor,factor:-factor,:]
    atlas_vol = skimage.measure.block_reduce(atlas_vol,(1, 2, 2, 2, 1), np.average)
    if atlas_vol.shape[1] > z_max:
        atlas_vol = atlas_vol[:, -z_max:, ...]
    holder = (1,z_max,256,256,1) - np.asarray(atlas_vol.shape)
    val_differences = [[i,0] for i in holder]
    atlas_vol = np.pad(atlas_vol, val_differences, 'constant', constant_values=(-1000))
    lower_threshold, upper_threshold = -75, 100
    atlas_vol = normalize(atlas_vol, lower_threshold, upper_threshold)
    vol_size = atlas_vol.shape[1:-1]
    train_generator = data_generator(atlas_vol,data_dir,lower_threshold=lower_threshold,upper_threshold=upper_threshold)
    x = train_generator.__getitem__(0)
    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    # random.shuffle(train_vol_names)  # shuffle volume list
    train_vol_names = glob.glob(os.path.join(data_dir, '*Registered_Data.npy'))
    assert len(train_vol_names) > 0, "Could not find any training data"

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    assert data_loss in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss in ['ncc', 'cc']:
        data_loss = losses.NCC().loss
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_desc = 'BatchNorm'
    tensorboard_output = os.path.join(model_dir,model_desc,'Tensorboard')
    model_output = os.path.join(model_dir, model_desc, 'Model_saves')
    if not os.path.isdir(tensorboard_output):
        os.makedirs(tensorboard_output)
    if not os.path.isdir(model_output):
        os.makedirs(model_output)

    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)


    # fit generator
    G = [1]
    gpu = 0
    if len(G) == 1:
        gpu = 0
    with K.tf.device('/gpu:' + str(gpu)):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        train_vol_names = glob.glob(os.path.join(data_dir, '*Registered_Data.npy'))
        train_example_gen = example_gen(train_vol_names, batch_size=batch_size)
        atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
        cvpr2018_gen = cvpr2018_gen(train_example_gen, atlas_vol_bs, batch_size=batch_size)
        # prepare the model
        # prepare the model
        # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, flow]
        # in the experiments, we use image_2 as atlas
        # model = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)
        model_class = Network_Building.new_model(image_size=vol_size, layers=layers, batch_normalization=False)
        model = model_class.model

        # load initial weights
        if load_model_file is not None:
            print('loading', load_model_file)
            model.load_weights(load_model_file)

        save_file_name = os.path.join(model_output,'weights-improvement-{epoch:02d}.hdf5')
        checkpoint = ModelCheckpoint(save_file_name, save_weights_only=False, period=1)
        tensorboard = TensorBoard(log_dir=tensorboard_output, batch_size=2, write_graph=True, write_grads=False,
                                  write_images=True, update_freq='epoch', histogram_freq=0)
        # Save_History,checkpoint,
        callbacks = [checkpoint, tensorboard]
        # save first iteration
        # model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))


        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)

        # single-gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            mg_model = model

        # compile
        model.compile(optimizer=Adam(lr=lr),
                         loss=[data_loss, losses.Grad('l2').loss],
                         loss_weights=[1.0, reg_param])

        # fit
        model.fit_generator(train_generator,
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=callbacks,
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)

if __name__ == "__main__":
    layers = {'Layer_0':{'Encoding':[16],'Decoding':[16,16]},
              'Layer_1': {'Encoding': [32, 64], 'Decoding': [32, 32]},
              'Base':{'Encoding':[64]}}
    # train_model(data_dir='../Reg_Data',atlas_file='../Reg_Data/Atlas_Data.npy',layers=layers,model_dir='../models/',gpu_id='0',
    #       lr=0.001,nb_epochs=10,reg_param=0.01,steps_per_epoch=30,batch_size=1,data_loss='mse',load_model_file=None)
