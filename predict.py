from utils import *
import time
import tensorflow as tf


def predict(model_path, test_files, segy_filename, inp_res, save_location, facies_names):
    if type(segy_filename) is list:
        segy_filename = segy_filename[0]

    # Make a master segy object
    segy_obj = segy_read(segy_file=segy_filename,
                           mode='create',
                           read_direc='full',
                           inp_res=inp_res)

    segy_obj.cube_num = 1
    segy_obj.data = np.expand_dims(segy_obj.data, axis=len(segy_obj.data.shape))

    model = tf.keras.models.load_model(model_path)

    n_maxpool = 0
    for l in model.layers:
        if type(l) is tf.keras.layers.MaxPool2D:
            n_maxpool += 1
    n_maxpool = n_maxpool // 2

    test_points = convert(test_files, facies_names)
    test_seis_data, test_masks, test_labels = points_to_images(test_points, segy_obj)

    out_x_shape = test_seis_data.shape[1]
    out_y_shape = test_seis_data.shape[2]
    for i in range(n_maxpool):
        out_x_shape = out_x_shape // 2
        out_y_shape = out_y_shape // 2

    for i in range(n_maxpool):
        out_x_shape = int(out_x_shape * 2)
        out_y_shape = int(out_y_shape * 2)

    in_x_shape = test_seis_data.shape[1]
    in_y_shape = test_seis_data.shape[2]

    diff_x = in_x_shape - out_x_shape
    diff_y = in_y_shape - out_y_shape
    test_seis_data = test_seis_data[:, :in_x_shape - diff_x, :in_y_shape - diff_y, :]
    test_masks = test_masks[:, :in_x_shape - diff_x, :in_y_shape - diff_y, :]
    test_labels = test_labels[:, :in_x_shape - diff_x, :in_y_shape - diff_y]

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    acc_file = open(save_location + 'acc.txt', 'w+')
    runtime_file = open(save_location + 'pred_time.txt', 'w+')

    binary_mask_predict = np.ones(test_seis_data.shape)

    start = time.time()
    prediction_prob = model.predict([test_seis_data, binary_mask_predict])
    prediction_time = time.time() - start

    for k in range(prediction_prob.shape[-1]):
        plt.imsave(save_location + f'prob_{k}.jpeg', np.transpose(prediction_prob[0, ..., k]), vmin=0, vmax=1)

    prediction = np.argmax(prediction_prob, axis=-1)
    plt.imsave(save_location + f'pred.jpeg', np.transpose(prediction[0]))
    runtime_file.write(f"{prediction_time}\n")

    labeled_rows, labeled_cols = np.where(test_masks[0, ..., 0] == 1)
    true_test_labels = test_labels[0, labeled_rows, labeled_cols]

    pred_test_labels = prediction[0, labeled_rows, labeled_cols]
    test_acc = np.sum(pred_test_labels == true_test_labels) / len(true_test_labels)
    acc_file.write(f"{test_acc}")