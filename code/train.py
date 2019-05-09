import os, glob
import sys
sys.path.append(os.path.join('..','ext','neuron'))
from Zip_data import Unzip_class
from Utils import normalize, visualize_model, create_model, train, load_atlas, data_generator, plot_scroll_Image, np

data_dir = os.path.join('..','Reg_Data')
if not os.path.exists(data_dir):
    print('Unzipping data')
    Unzip_class(os.path.join('..','Reg_Data'),r'..')
    print('Finished unzipping')
model_dir = os.path.join('..','models')

atlas_file= os.path.join('..','reg_data','Atlas_Data.npy')
atlas_vol = load_atlas(atlas_file)

layers = {'Layer_0':{'Encoding':[16,32],'Decoding':[32,16,8]},
          'Base':{'Encoding':[64]}}
model_desc = 'Shallow_net' # Name of your model
# The numbers inside are the number of filter banks, you can have mulitple filter banks per layer

train_generator = data_generator(atlas_vol,data_dir)
print('We have ' + str(len(train_generator)) + ' registrations available')
Moving_names = glob.glob(r'K:\Morfeus\AAPM_SummerSchool\voxelmorph_all_data\*Moving_Data.npy')
for name in Moving_names:
    data = np.load(name)
    data[data>100] = 100
    data[data<-50] = -50
    xxx = 1
learning_rate = 0.001 # Rate at which our gradients will change during each back propogation, typically in range of 1e-2 to 1e-5
number_of_epochs = 10 # The number of epochs to be trained, one epoch means that you have seen the entirety of your dataset
                      # However, since we defined steps per epoch this might not apply
regularization_parameter = 0.01 # Lambda in regularization equation
steps_per_epoch = 10
loss_function = 'mse'
batch_normalization = True

model, callbacks = create_model(layers, atlas_vol.shape[1:-1], model_desc, batch_norm=batch_normalization, data_generator=train_generator)

train(model, train_generator, callbacks, learning_rate, number_of_epochs,
      regularization_parameter, steps_per_epoch,loss_function)
