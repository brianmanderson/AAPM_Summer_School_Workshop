import sys
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, Concatenate, BatchNormalization
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal


class UNet_Core():
    def __init__(self, activation='elu', alpha=0.2, input_size=(176,512,512), visualize=False,batch_normalization=True):
        self.filters = (3,3,3)
        self.activation = activation
        self.alpha = alpha
        self.pool_size = (2,2,2)
        self.input_size = input_size
        self.visualize=visualize
        self.batch_normalization = batch_normalization

    def conv_block(self,output_size,x, strides=1):
        x = Conv3D(output_size, self.filters, activation=None, padding='same',
                   name=self.desc, kernel_initializer='he_normal', strides=strides)(x)
        if self.activation != 'LeakyReLU':
            x = LeakyReLU(self.alpha)(x)
        else:
            x = Activation(self.activation)(x)
        self.layer += 1
        if not self.visualize and self.batch_normalization:
            x = BatchNormalization()(x)
        return x

    def get_unet(self, layers_dict):
        atlas = Input(shape=self.input_size + (1,), name='Atlas')
        moving = Input(shape=self.input_size + (1,), name='Moving')
        x = Concatenate(name='Input_concat')([atlas, moving])
        self.layer = 0
        layer_vals = {}
        self.desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in layers_dict:
            if layer == 'Base':
                continue
            layer_order.append(layer)
            all_filters = layers_dict[layer]['Encoding']
            for i in range(len(all_filters)):
                strides = 2 if i == len(all_filters)-1 else 1
                self.desc = layer + '_Encoding_Conv' + str(i) if strides == 1 else layer + '_Strided_Conv' + str(i)
                if strides == 2 and layer_index not in layer_vals:
                    layer_vals[layer_index] = x
                x = self.conv_block(all_filters[i], x, strides=strides)
                if strides == 1 or layer_index not in layer_vals:
                    layer_vals[layer_index] = x
            layer_index += 1
        if 'Base' in layers_dict:
            strides = 1
            all_filters = layers_dict['Base']['Encoding']
            for i in range(len(all_filters)):
                self.desc = 'Base_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, strides=strides)
        self.desc = 'Decoder'
        self.layer = 0
        layer_order.reverse()
        for layer in layer_order:
            layer_index -= 1
            all_filters = layers_dict[layer]['Decoding']
            x = UpSampling3D(size=(2, 2, 2), name='Upsampling' + str(self.layer) + '_UNet')(x)
            x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, layer_vals[layer_index]])
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x)
        model = Model(inputs=[atlas, moving], outputs=x)
        self.created_model = model


class new_model(object):
    def __init__(self, layers, image_size=(56,256,256,1), indexing='ij',batch_normalization=False,visualize=False):
        self.indexing = indexing
        UNet_Core_class = UNet_Core(input_size=image_size, batch_normalization=batch_normalization, visualize=visualize)
        UNet_Core_class.get_unet(layers)
        self.unet_model = UNet_Core_class.created_model
        self.make_flow()

    def make_flow(self):
        [Atlas, Moving] = self.unet_model.inputs
        x = self.unet_model.output

        flow = Conv3D(3, kernel_size=3, padding='same', name='flow',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=self.indexing)([Atlas, flow])
        # prepare model
        self.model = Model(inputs=[Atlas, Moving], outputs=[y, flow])


if __name__ == '__main__':
    None