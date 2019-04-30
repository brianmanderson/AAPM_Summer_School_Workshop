import os, glob
import skimage.measure
import SimpleITK as sitk
import numpy as np
from Utils import plot_scroll_Image
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def normalize(X, lower, upper):
    X[X<lower] = lower
    X[X > upper] = upper
    X = (X - lower)/(upper - lower)
    return X


def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))


def register_test_images(fixed,moving, fixed_spacing_info=(3,.9875,0.9875), moving_spacing_info=(5,.9875,.9875)):
    fixed = sitk.GetImageFromArray(np.squeeze(fixed).astype('float32'))
    moving = sitk.GetImageFromArray(np.squeeze(moving).astype('float32'))

    moving.SetSpacing(moving_spacing_info)
    fixed.SetSpacing(fixed_spacing_info)
    numberOfBins = 24
    samplingPercentage = 0.10


    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 500)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    # simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    # simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    # cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    # sitk.Show(cimg, "ImageRegistration4 Composition")
    return sitk.GetArrayFromImage(out)


class register_class(object):

    def __init__(self, atlas, slice_info_atlas):
        self.atlas = atlas
        self.slice_info_atlas = slice_info_atlas

    def register_moving(self, train_name):
        atlas = self.atlas
        slice_info_atlas = self.slice_info_atlas
        data = np.load(train_name)
        while data.shape[2] > atlas.shape[2]:
            data = skimage.measure.block_reduce(data, (1, 2, 2, 2, 1), np.average)
        data = normalize(data, -1000, 1000)
        fid = open(train_name.replace('.npy', '.txt'))
        slice_info = fid.readline()
        fid.close()
        slice_info = slice_info.split(',')
        slice_info = [float(i) for i in slice_info if i != '']
        if len(slice_info) == 2:
            slice_info = [3] + slice_info
        slice_info = tuple(slice_info)
        registered = register_test_images(atlas, data, slice_info_atlas, slice_info)
        registered = registered*2000 - 1000
        np.save(train_name.replace('Moving', 'Registered'), registered[None, ..., None])
        return None


def main(path):
    atlas_name = os.path.join(path,'Atlas_Data.npy')
    atlas = np.load(atlas_name)
    atlas = normalize(atlas, -1000,1000)
    atlas = skimage.measure.block_reduce(atlas, (1, 2, 2, 2, 1), np.average)
    fid = open(atlas_name.replace('.npy', '.txt'))
    slice_info_atlas = fid.readline()
    fid.close()
    slice_info_atlas = slice_info_atlas.split(',')
    slice_info_atlas = tuple([float(i) for i in slice_info_atlas if i != ''])
    train_vol_names = glob.glob(os.path.join(path, '*Moving_Data.npy'))
    reg_vol_names = glob.glob(os.path.join(path, '*Registered_Data.npy'))
    train_vol_names = [i for i in train_vol_names if not os.path.exists(i.replace('Moving_Data','Registered_Data'))]
    for_workers(atlas, slice_info_atlas, train_vol_names)
    return None

def worker_def(A):
    q, atlas, slice_info_atlas = A
    registration_object = register_class(atlas, slice_info_atlas)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                print('Running on ' + item)
                registration_object.register_moving(item)
            except:
                print('failed?')
            q.task_done()

def for_workers(atlas, slice_info_atlas, train_vol_names):
    thread_count = cpu_count() - 1  # Leaves you one thread for doing things with
    print('This is running on ' + str(thread_count) + ' threads')
    q = Queue(maxsize=thread_count)
    A = [q, atlas, slice_info_atlas]
    register_class_van = register_class(atlas, slice_info_atlas)
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(A,))
        t.start()
        threads.append(t)
    for train_name in train_vol_names:
        # register_class_van.register_moving(train_name)
        q.put(train_name)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
if __name__ == '__main__':
    main(r'K:\Morfeus\AAPM_SummerSchool\voxelmorph_all_data')