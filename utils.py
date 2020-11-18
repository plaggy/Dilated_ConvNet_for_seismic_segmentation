import numpy as np
import segyio
import matplotlib.pyplot as plt
import copy
import seaborn as sn
from skimage.util import random_noise
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder


def process_coords(points_coords, seis_spec, cube_incr_x, cube_incr_y, cube_incr_z, cube_step_interval, shuffle=True):
    coords_copy = points_coords
    if shuffle == True:
        np.random.shuffle(coords_copy)

    inline_start = seis_spec.inl_start
    inline_step = seis_spec.inl_step
    xline_start = seis_spec.xl_start
    xline_step = seis_spec.xl_step
    t_start = seis_spec.t_start
    t_step = seis_spec.t_step

    coords_copy_norm = np.array([(coords_copy[:, 0] - inline_start) // inline_step + cube_incr_x * cube_step_interval,
                                 (coords_copy[:, 1] - xline_start) // xline_step + cube_incr_y * cube_step_interval,
                                 (coords_copy[:, 2] - t_start) // t_step + cube_incr_z,
                                 coords_copy[:, 3]]).T

    return coords_copy_norm


def segy_read(segy_file, mode, scale=1, inp_cube=None, read_direc='xline', inp_res=np.float32):

    if mode == 'create':
        print('Starting SEG-Y decompressor')
        output = segyio.spec()

    elif mode == 'add':
        if inp_cube is None:
            raise ValueError('if mode is add inp_cube must be provided')
        print('Starting SEG-Y adder')
        cube_shape = inp_cube.shape
        data = np.empty(cube_shape[0:-1])

    else:
        raise ValueError('mode must be create or add')

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r") as segyfile:
        segyfile.mmap()

        if mode == 'create':
            # Store some initial object attributes
            output.inl_start = segyfile.ilines[0]
            output.inl_end = segyfile.ilines[-1]
            output.inl_step = segyfile.ilines[1] - segyfile.ilines[0]

            output.xl_start = segyfile.xlines[0]
            output.xl_end = segyfile.xlines[-1]
            output.xl_step = segyfile.xlines[1] - segyfile.xlines[0]

            output.t_start = int(segyfile.samples[0])
            output.t_end = int(segyfile.samples[-1])
            output.t_step = int(segyfile.samples[1] - segyfile.samples[0])


            # Pre-allocate a numpy array that holds the SEGY-cube
            data = np.empty((segyfile.xline.length,segyfile.iline.length,\
                            (output.t_end - output.t_start)//output.t_step+1), dtype = np.float32)

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for il_index in range(segyfile.xline.len):
                data[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]
            #print(end - start)

        elif read_direc == 'xline':
            #start = time.time()
            for xl_index in range(segyfile.iline.len):
                data[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'full':
            #start = time.time()
            data = segyio.tools.cube(segy_file)
            #end = time.time()
            #print(end - start)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')

        factor = scale/np.amax(np.absolute(data))
        if inp_res == np.float32:
            data = (data*factor)
        else:
            data = (data*factor).astype(dtype = inp_res)

    if mode == 'create':
        output.data = data
        return output
    else:
        return output


def points_to_images(points, segy_obj):
    if points[0][0] == points[1][0]:
        il = points[0][0]
        mask = np.zeros((1, segy_obj.data.shape[1], segy_obj.data.shape[2], 1))
        label = np.zeros((1, segy_obj.data.shape[1], segy_obj.data.shape[2]))

        in_idx = (points[:, 2] > segy_obj.t_start) & (points[:, 2] < segy_obj.t_end)
        points = points[in_idx]
        points_xl = (points[:, 1] - segy_obj.xl_start) // segy_obj.xl_step
        points_t = (points[:, 2] - segy_obj.t_start) // segy_obj.t_step
        mask[0, points_xl, points_t, 0] = 1

        for i in range(np.unique(points[:, 3]).shape[0]):
            idx = np.where(points[:, 3] == i)[0]
            label[0, points[idx, 1] - segy_obj.xl_start, (points[idx, 2] - segy_obj.t_start) // segy_obj.t_step] = i

        seis_data = segy_obj.data[il - segy_obj.inl_start, :, :, :][np.newaxis, :, :, :]

    elif points[0][1] == points[1][1]:
        xl = points[0][1]
        mask = np.zeros((1, segy_obj.data.shape[0], segy_obj.data.shape[2], 1))
        label = np.zeros((1, segy_obj.data.shape[0], segy_obj.data.shape[2]))

        in_idx = (points[:, 2] > segy_obj.t_start) & (points[:, 2] < segy_obj.t_end)
        points = points[in_idx]
        points_il = (points[:, 0] - segy_obj.inl_start) // segy_obj.inl_step
        points_t = (points[:, 2] - segy_obj.t_start) // segy_obj.t_step
        mask[0, points_il, points_t, 0] = 1

        for i in range(np.unique(points[:, 3]).shape[0]):
            idx = np.where(points[:, 3] == i)[0]
            label[0, points[idx, 0] - segy_obj.inl_start, (points[idx, 2] - segy_obj.t_start) // segy_obj.t_step] = i

        seis_data = segy_obj.data[:, xl - segy_obj.xl_start, :, :][np.newaxis, :, :, :]

    else:
        raise ValueError("points' format is not recognized")

    return seis_data, mask, label


def convert(file_list, facies_names):
    # preallocate space for the adr_list, the output containing all the adresses and classes
    adr_list = np.empty([0, 4], dtype=np.int32)

    file_list_by_facie = []
    for facie in facies_names:
        facie_list = []
        for filename in file_list:
            if facie in filename:
                facie_list.append(filename)
        file_list_by_facie.append(facie_list)

    # Itterate through the list of example adresses and store the class as an integer
    for i, files in enumerate(file_list_by_facie):
        for filename in files:
            a = np.loadtxt(filename, skiprows=0, usecols=range(3), dtype=np.int32)
            adr_list = np.append(adr_list, np.append(a, i*np.ones((len(a), 1), dtype=np.int32), axis=1), axis=0)

    return adr_list


def simple_patches_generator(seismic_data, masks, labels, window_size, window_step, batch_size):
    while True:
        for k, (line, mask, label) in enumerate(zip(seismic_data, masks, labels)):
            window_start_x = 0
            window_end_x = window_size
            window_start_y = 0
            window_end_y = window_size
            line = line[0]
            mask = mask[0]
            label = label[0]

            n_steps_x = (line.shape[0] - window_size) // window_step + 1
            n_steps_y = (line.shape[1] - window_size) // window_step + 1

            n_steps_per_batch = int(np.ceil(batch_size / 3))
            seis_patches = np.zeros((n_steps_per_batch * 3, window_size, window_size, 1))
            mask_patches = np.zeros((n_steps_per_batch * 3, window_size, window_size, 1))
            label_patches = np.zeros((n_steps_per_batch * 3, window_size, window_size))
            last_batch_steps = (n_steps_x * n_steps_y) % n_steps_per_batch

            if last_batch_steps > 0:
                seis_last_batch = np.zeros((last_batch_steps * 3, window_size, window_size, 1))
                mask_last_batch = np.zeros((last_batch_steps * 3, window_size, window_size, 1))
                label_last_batch = np.zeros((last_batch_steps * 3, window_size, window_size))
                n_steps_done_last_batch = 0
            n_batches_yielded = 0

            for i in range(n_steps_x * n_steps_y):
                seis_ex = line[window_start_x:window_end_x, window_start_y:window_end_y, :]
                flipped_seis = tf.image.flip_up_down(seis_ex)
                noised_seis = random_noise(seis_ex, mode='gaussian', var=0.001)
                binary_mask_ex = mask[window_start_x:window_end_x, window_start_y:window_end_y, :]
                flipped_mask = tf.image.flip_up_down(binary_mask_ex)
                classes_ex = label[window_start_x:window_end_x, window_start_y:window_end_y]
                flipped_classes = tf.image.flip_up_down(classes_ex[:, :, np.newaxis])

                if n_steps_x * n_steps_y - (i + 1) < last_batch_steps and last_batch_steps > 0:
                    n_steps_done_last_batch += 1
                    seis_last_batch[(i % n_steps_per_batch) * 3] = seis_ex
                    mask_last_batch[(i % n_steps_per_batch) * 3] = binary_mask_ex
                    label_last_batch[(i % n_steps_per_batch) * 3] = classes_ex
                    seis_last_batch[(i % n_steps_per_batch) * 3 + 1] = flipped_seis
                    mask_last_batch[(i % n_steps_per_batch) * 3 + 1] = flipped_mask
                    label_last_batch[(i % n_steps_per_batch) * 3 + 1] = flipped_classes[:, :, 0]
                    seis_last_batch[(i % n_steps_per_batch) * 3 + 2] = noised_seis
                    mask_last_batch[(i % n_steps_per_batch) * 3 + 2] = binary_mask_ex
                    label_last_batch[(i % n_steps_per_batch) * 3 + 2] = classes_ex
                else:
                    seis_patches[(i % n_steps_per_batch) * 3] = seis_ex
                    mask_patches[(i % n_steps_per_batch) * 3] = binary_mask_ex
                    label_patches[(i % n_steps_per_batch) * 3] = classes_ex
                    seis_patches[(i % n_steps_per_batch) * 3 + 1] = flipped_seis
                    mask_patches[(i % n_steps_per_batch) * 3 + 1] = flipped_mask
                    label_patches[(i % n_steps_per_batch) * 3 + 1] = flipped_classes[:, :, 0]
                    seis_patches[(i % n_steps_per_batch) * 3 + 2] = noised_seis
                    mask_patches[(i % n_steps_per_batch) * 3 + 2] = binary_mask_ex
                    label_patches[(i % n_steps_per_batch) * 3 + 2] = classes_ex

                if i == n_steps_x * n_steps_y - 1:
                    if last_batch_steps > 0:
                        yield [seis_last_batch, mask_last_batch], label_last_batch
                    else:
                        yield [seis_patches, mask_patches], label_patches
                else:
                    if (i + 1) % n_steps_per_batch == 3:
                        n_batches_yielded += 1
                        yield [seis_patches, mask_patches], label_patches

                if window_end_x + window_step > line.shape[0]:
                    window_start_x = 0
                    window_end_x = window_size
                    window_start_y += window_step
                    window_end_y += window_step
                else:
                    window_start_x += window_step
                    window_end_x += window_step


def patches_creator(seismic_data, masks, labels, window_size, window_step):
    seis_patches = []
    mask_patches = []
    label_patches = []

    for line, mask, label in zip(seismic_data, masks, labels):
        window_start_x = 0
        window_end_x = window_size
        window_start_y = 0
        window_end_y = window_size
        n_steps_x = (line.shape[0] - window_size) // window_step + 1
        n_steps_y = (line.shape[1] - window_size) // window_step + 1

        for i in range(n_steps_x * n_steps_y):
            seis_ex = line[window_start_x:window_end_x, window_start_y:window_end_y, :]
            seis_patches.append(seis_ex)
            binary_mask_ex = mask[window_start_x:window_end_x, window_start_y:window_end_y, :]
            mask_patches.append(binary_mask_ex)
            classes_ex = label[window_start_x:window_end_x, window_start_y:window_end_y]
            label_patches.append(classes_ex)

            seis_patches.append(np.flipud(seis_ex))
            mask_patches.append(np.flipud(binary_mask_ex))
            label_patches.append(np.flipud(classes_ex))

            seis_patches.append(random_noise(seis_ex, mode='gaussian', var=0.001))
            mask_patches.append(binary_mask_ex)
            label_patches.append(classes_ex)

            if window_end_x + window_step > line.shape[0]:
                window_start_x = 0
                window_end_x = window_size
                window_start_y += window_step
                window_end_y += window_step
            else:
                window_start_x += window_step
                window_end_x += window_step

    return np.array(seis_patches), np.array(mask_patches), np.array(label_patches)


def generate_coordinates(section_number, section_type, segy_obj):
    seis_arr = segy_obj.data

    if section_type == 'xline':
        first_il = segy_obj.il_start
        t_step = segy_obj.t_step
        first_t = segy_obj.t_start // t_step
        n_il = seis_arr.shape[0]
        n_ts = seis_arr.shape[2]
        predict_coord = []
        for il in range(first_il, segy_obj.xl_start + n_il):
            for ts in range(first_t, first_t + n_ts):
                predict_coord.append([il, section_number, ts * t_step, 0])

    elif section_type == 'inline':
        first_xl = segy_obj.xl_start
        t_step = segy_obj.t_step
        first_t = segy_obj.t_start // t_step
        n_xl = seis_arr.shape[1]
        n_ts = seis_arr.shape[2]
        predict_coord = []
        for xl in range(first_xl, segy_obj.xl_start + n_xl):
            for ts in range(first_t, first_t + n_ts):
                predict_coord.append([section_number, xl, ts * t_step, 0])

    else:
        raise ValueError('section_type must be inline or xline')

    return np.array(predict_coord)


def save_test_prediction(section_number, section_type, test_coords_prep, predictions_test, cube_incr_x, cube_incr_y,
                         cube_incr_z, cube_step_interval, segy_filename, segy_obj, write_location, mode):
    output_file = write_location + 'test_prediction.sgy'

    test_coords = generate_coordinates(section_number, section_type, segy_obj)
    test_coords = process_coords(test_coords, segy_obj, cube_incr_x, cube_incr_y, cube_incr_z,
                                            cube_step_interval, shuffle=False)

    test_coords_prep_loc = copy.deepcopy(test_coords_prep)
    test_coords_prep_loc[:, 0] -= cube_incr_x * cube_step_interval
    test_coords_prep_loc[:, 1] -= cube_incr_y * cube_step_interval
    test_coords_prep_loc[:, 2] -= cube_incr_z * cube_step_interval
    ind = np.lexsort((test_coords[:, 2], test_coords[:, 1], test_coords[:, 0]))
    test_coords_prep_loc = test_coords_prep_loc[ind]
    predictions_test = predictions_test[ind]

    with segyio.open(segy_filename) as src:
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = src.samples
        spec.ilines = np.unique(test_coords[:, 0])
        spec.xlines = np.unique(test_coords[:, 1])
        with segyio.create(output_file, spec) as dst:
            dst.text[0] = src.text[0]
            for i, iline in enumerate((spec.ilines - segy_obj.inl_start) // segy_obj.inl_step):
                for j, xline in enumerate((spec.xlines - segy_obj.xl_start) // segy_obj.xl_step):
                    if mode == 'xline':
                        tr_idx = i
                    if mode == 'iline':
                        tr_idx = j
                    dst.header[tr_idx].update(src.header[iline * len(src.xlines) + xline])
                    dst.trace[tr_idx] = np.ones(len(src.samples)) * 9
                    idx = np.all(np.array([test_coords_prep_loc[:, 0] == iline, test_coords_prep_loc[:, 1] == xline]),
                                 axis=0)
                    samples = test_coords_prep_loc[idx][:, 2]
                    trace_vals = dst.trace[tr_idx]
                    trace_vals[samples] = predictions_test[idx]
                    dst.trace[tr_idx] = trace_vals


def printout(history, write_location, y_test, pred_test, facies_list):
    facies_list = ['background'] + facies_list[1:]

    if not os.path.exists(write_location):
        os.makedirs(write_location)

    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('batch #')
    ax1.set_ylabel('loss', color='blue')
    ax1.plot(np.array(history.batch_loss), color='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('acc', color='red')
    ax2.plot(np.array(history.batch_acc), color='red')
    plt.savefig(write_location + 'batch_history.jpg')

    np.savetxt(write_location + 'history_batch_loss.txt', np.array(history.batch_loss))
    np.savetxt(write_location + 'history_batch_acc.txt', np.array(history.batch_acc))
    np.savetxt(write_location + 'history_val_acc.txt', np.array(history.val_acc))
    np.savetxt(write_location + 'history_val_loss.txt', np.array(history.val_loss))

    plt.figure()
    cm = confusion_matrix(y_test.astype(int), pred_test.astype(int), normalize='true')
    print(facies_list)
    print(cm)
    df_cm = pd.DataFrame(cm, index=[facies_list[i] for i in list(np.arange(cm.shape[0]))],
                         columns=[facies_list[i] for i in list(np.arange(cm.shape[0]))])
    sn.heatmap(df_cm, annot=True, fmt='.2%', cmap='Blues')
    plt.savefig(write_location + "confusion_matrix.jpg")

    enc = OneHotEncoder(sparse=False)
    categories = np.arange(0, len(facies_list)).reshape(-1, 1)
    enc.fit(categories)
    y_test = enc.transform(y_test.reshape(-1, 1))
    pred_test = enc.transform(pred_test.reshape(-1, 1))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if not os.path.exists(write_location + 'ROC_curves/'):
        os.makedirs(write_location + 'ROC_curves/')
    for i in range(pred_test.shape[1]):
        if np.sum(y_test[:, i]) == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.6f)' % roc_auc[i])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for a class %d' % i)
        plt.legend(loc="lower right")
        plt.savefig(write_location + 'ROC_curves/curve_%d' % i)