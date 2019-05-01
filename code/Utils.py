import matplotlib.pyplot as plt
import os
# third-party imports
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend as K
import Network_Building
import losses
import numpy as np
import skimage.measure


def normalize(X, lower, upper):
    X[X<lower] = lower
    X[X > upper] = upper
    X = (X - lower)/(upper - lower)
    return X


def plot_scroll_Image(x):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, x.astype('float32'))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def visualize_model(layers, vol_size, model_desc):
    K.clear_session()
    model_class = Network_Building.new_model(image_size=vol_size,layers=layers,
                                             visualize=True, batch_normalization=False)
    model = model_class.model
    tensorboard_output = os.path.join('..','Tensorboard_models',model_desc)
    if not os.path.exists(tensorboard_output):
        os.makedirs(tensorboard_output)
    tensorboard = TensorBoard(log_dir=tensorboard_output, batch_size=2, write_graph=True, write_grads=False,
                              write_images=True, update_freq='epoch', histogram_freq=0)
    tensorboard.set_model(model)
    tensorboard._write_logs({},0)
    print('Model created at: ' + os.path.abspath(tensorboard_output))
    return None


def create_model(layers, vol_size, model_desc, batch_norm=False):
    model_dir = os.path.join('..', 'models')
    K.clear_session()
    model_class = Network_Building.new_model(image_size=vol_size,layers=layers,
                                             visualize=False, batch_normalization=batch_norm)
    model = model_class.model
    tensorboard_output = os.path.join('..','Tensorboard_models',model_desc)
    if not os.path.exists(tensorboard_output):
        os.makedirs(tensorboard_output)
    tensorboard = TensorBoard(log_dir=tensorboard_output, batch_size=2, write_graph=True, write_grads=False,
                              write_images=True, update_freq='epoch', histogram_freq=0)
    model_output = os.path.join(model_dir, model_desc, 'Model_saves')
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    save_file_name = os.path.join(model_output,'weights-improvement-{epoch:02d}.hdf5')
    checkpoint = ModelCheckpoint(save_file_name, save_weights_only=False, period=1)
    callbacks = [checkpoint, tensorboard]
    return model, callbacks


def train(model, train_generator, callbacks, learning_rate, number_of_epochs,
          regularization_parameter, steps_per_epoch,loss_function='mse'):
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    K.set_session(sess)
    model.compile(optimizer=Adam(lr=learning_rate),
                     loss=[loss_function, losses.Grad('l2').loss],
                     loss_weights=[1.0, regularization_parameter])

    model.fit_generator(train_generator,initial_epoch=0,epochs=number_of_epochs,
                        callbacks=callbacks,steps_per_epoch=steps_per_epoch,verbose=1)

def load_atlas(atlas_file):
    atlas_vol = np.load(atlas_file) #['vol'][np.newaxis, ..., np.newaxis]
    z_max = 64
    atlas_vol = skimage.measure.block_reduce(atlas_vol,(1, 2, 2, 2, 1), np.average)
    if atlas_vol.shape[1] > z_max:
        atlas_vol = atlas_vol[:, -z_max:, ...]
    holder = (1,z_max,256,256,1) - np.asarray(atlas_vol.shape)
    val_differences = [[i,0] for i in holder]
    atlas_vol = np.pad(atlas_vol, val_differences, 'constant', constant_values=(-1000))
    lower_threshold, upper_threshold = -75, 100
    atlas_vol = normalize(atlas_vol, lower_threshold, upper_threshold)
    return atlas_vol


if __name__ == '__main__':
    None