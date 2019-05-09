import matplotlib.pyplot as plt
import os, glob, copy
# third-party imports
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import Sequence
import keras.backend as K
import Network_Building
import losses
import numpy as np
import skimage.measure
from PIL import Image
import io



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
        if X.shape[2] > self.atlas.shape[2]:
            X = skimage.measure.block_reduce(X, (1, 2, 2, 2, 1), np.average)
        if X.shape[1] > int(z_max):
            X = X[:,-z_max:,...]
        while X.shape[2] > self.atlas.shape[2]:
            X = skimage.measure.block_reduce(X, (1, 2, 2, 2, 1), np.average)
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


def normalize(X, lower, upper):
    X[X<lower] = lower
    X[X > upper] = upper
    X = (X - lower)/(upper - lower)
    return X


def plot_scroll_Image(x):
    '''
    :param x: input to view of form [rows, columns, # images]
    :return:
    '''
    if x.dtype not in ['float32','float64']:
        x = copy.deepcopy(x).astype('float32')
    if len(x.shape) > 3:
        x = np.squeeze(x)
    if len(x.shape) == 3:
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x,[1,2,0])
    fig, ax = plt.subplots(1, 1)
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=0)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker
    #Image is input in the form of [#images,512,512,#channels]

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



class TensorBoardImage(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch', tag='', data_generator=None):
        super().__init__(log_dir=log_dir,
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch')
        self.tag = tag
        self.log_dir = log_dir
        self.data_generator = data_generator
        if self.data_generator:
            x,_ = self.data_generator.__getitem__(0)
            self.images, self.rows, self.cols = x[0].shape[1], x[0].shape[2], x[0].shape[3]
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.embeddings_data = embeddings_data
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def make_image(self, tensor):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        tensor = np.squeeze(tensor)
        height, width = tensor.shape
        tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor)) * 255
        image = Image.fromarray(tensor.astype('uint8'))
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=1,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.data_generator:
            self.add_images(epoch)
        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {_input: embeddings_data[idx][batch]
                                     for idx, _input in enumerate(self.model.input)}
                    else:
                        feed_dict = {self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir,
                                                 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def add_images(self, epoch):
        # Load image
        self.data_generator.shuffle()
        num_images = min([5,len(self.data_generator)])
        out_atlas, out_moving, out_deformed = np.ones([self.rows, int(self.cols*num_images+int(self.cols/10)*(num_images-1))]), \
                                        np.ones([self.rows, int(self.cols * num_images + int(self.cols/10) * (num_images - 1))]), \
                                        np.ones([self.rows, int(self.cols * num_images + int(self.cols/10) * (num_images - 1))])
        step = self.cols
        for i in range(num_images):
            start = int(50*i)
            inputs, _ = self.data_generator.__getitem__(i)
            pred = self.model.predict_on_batch(inputs)[0]
            out_atlas[:,step*i + start:step*(i+1)+start] = inputs[0][0,int(self.images/2),...,0]
            out_moving[:, step * i + start:step * (i + 1) + start] = inputs[1][0,int(self.images/2),...,0]
            out_deformed[:, step * i + start:step * (i + 1) + start] = pred[0,int(self.images/2),...,0]
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Atlas', image=self.make_image(out_atlas))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Moving', image=self.make_image(out_moving))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Deformed', image=self.make_image(out_deformed))])
        self.writer.add_summary(summary, epoch)
        return None


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


def create_model(layers, vol_size, model_desc, batch_norm=False, data_generator=None):
    model_dir = os.path.join('..', 'models')
    K.clear_session()
    model_class = Network_Building.new_model(image_size=vol_size,layers=layers,
                                             visualize=False, batch_normalization=batch_norm)
    model = model_class.model
    tensorboard_output = os.path.join('..','Tensorboard_models',model_desc)
    if not os.path.exists(tensorboard_output):
        os.makedirs(tensorboard_output)
    tensorboard = TensorBoardImage(log_dir=tensorboard_output, batch_size=2, write_graph=True, write_grads=False,
                              write_images=True, update_freq='epoch', histogram_freq=0, data_generator=data_generator)
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

def load_atlas(atlas_file, reduction_factor=1):
    atlas_vol = np.load(atlas_file) #['vol'][np.newaxis, ..., np.newaxis]
    z_max = 128
    out_size = 512
    if reduction_factor > 0:
        atlas_vol = skimage.measure.block_reduce(atlas_vol,(1, 2, 2, 2, 1), np.average)
        reduction_factor -= 1
        z_max //= 2
        out_size //= 2
    if atlas_vol.shape[1] > z_max:
        atlas_vol = atlas_vol[:, -z_max:, ...]
    holder = (1,z_max,out_size,out_size,1) - np.asarray(atlas_vol.shape)
    val_differences = [[i,0] for i in holder]
    if np.max(val_differences) > 0:
        atlas_vol = np.pad(atlas_vol, val_differences, 'constant', constant_values=(-1000))
    for i in range(reduction_factor):
        atlas_vol = skimage.measure.block_reduce(atlas_vol, (1, 2, 2, 2, 1), np.average)
    lower_threshold, upper_threshold = -75, 100
    atlas_vol = normalize(atlas_vol, lower_threshold, upper_threshold)
    return atlas_vol


if __name__ == '__main__':
    None