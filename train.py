import time

from utils import *

tf.config.experimental_run_functions_eagerly(True)


def build_model(data, n_conv_layers, num_dilate_layers, n_filters, kernel_size, n_classes,
                use_batchnorm=True, maxpool=True, telescopic=False, dropout=False):
    n_maxpool = 0
    n_upsample = 0
    max_n_maxpool = num_dilate_layers // 2
    data_input = tf.keras.Input(shape=(None, None, data.shape[3]), name='input')
    mask = tf.keras.Input(shape=(None, None, data.shape[3]), name='mask')

    x = tf.cast(data_input, tf.float32)
    m = tf.cast(mask, tf.float32)

    for i in range(n_conv_layers):
        x = tf.keras.layers.Multiply(name=f'conv_mult_{i}')([x, m])
        if telescopic:
            if n_filters < 64:
                n_filters *= 2
            if kernel_size > 3:
                kernel_size -= 2

        x = tf.keras.layers.Conv2D(n_filters, kernel_size, use_bias=False, padding='same', name=f'conv_{i}')(x)

        norm = tf.keras.layers.Conv2D(n_filters, kernel_size, trainable=False, use_bias=False, padding="same",
                                      kernel_initializer=tf.constant_initializer(1), name=f'norm_init_{i}')(m)
        norm = tf.where(tf.math.equal(norm, 0), tf.zeros_like(norm), tf.math.reciprocal(norm))
        _, _, _, bias_size = norm.get_shape()
        b = tf.Variable(tf.constant(0.0, shape=[bias_size]), trainable=True)
        x = tf.keras.layers.Multiply(name=f'conv_norm_{i}')([x, norm])
        x = tf.nn.bias_add(x, b)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if dropout:
            x = tf.keras.layers.Dropout(0.5)(x)
        if maxpool:
            if (i + 1) % 2 == 0 and n_maxpool < max_n_maxpool:
                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
                m = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(m)
                n_maxpool += 1

    # Add dilational conv2D layers
    dilate_factors = [2 ** (i + 1) for i in np.arange(num_dilate_layers)]

    for i in range(num_dilate_layers):
        d = dilate_factors[i]
        x = tf.keras.layers.Multiply()([x, m])
        if telescopic:
            if n_filters > 8:
                n_filters //= 2
            if kernel_size < 7:
                kernel_size += 2
        x = tf.keras.layers.Conv2D(n_filters, kernel_size, use_bias=False, padding='same', dilation_rate=(d, d))(x)
        norm = tf.keras.layers.Conv2D(n_filters, kernel_size, trainable=False, use_bias=False, padding="same",
                                      kernel_initializer=tf.constant_initializer(1))(m)
        norm = tf.where(tf.math.equal(norm, 0), tf.zeros_like(norm), tf.math.reciprocal(norm))
        _, _, _, bias_size = norm.get_shape()
        b = tf.Variable(tf.constant(0.0, shape=[bias_size]), trainable=True)
        x = tf.keras.layers.Multiply()([x, norm])
        x = tf.nn.bias_add(x, b)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if dropout:
            x = tf.keras.layers.Dropout(0.5)(x)
        if maxpool:
            if (i + 1) % 2 == 0 and n_upsample < n_maxpool:
                x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
                m = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(m)
                n_upsample += 1

    x = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(x)
    x = tf.keras.layers.Multiply()([x, m])
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("softmax")(x)
    x = tf.keras.layers.Multiply()([x, m])

    model = tf.keras.Model(inputs=[data_input, mask], outputs=x)

    return model


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_loss = []
        self.batch_acc = []
        self.val_acc = []
        self.val_loss = []

    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))
        self.batch_acc.append(logs.get('accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))


def train_wrapper(train_dict):
    train_files = train_dict['train_files']
    test_files = train_dict['test_files']
    segy_filename = train_dict['segy_filename']
    inp_res = train_dict['inp_res']
    n_epoch = train_dict['epochs']
    validation_split = train_dict['validation_split']
    save_location_global = train_dict['save_location']
    facies_names = train_dict['facies_names']
    num_conv_layers_samples = train_dict['num_conv_layers_samples']
    num_dil_layers_samples = train_dict['num_dil_layers_samples']
    kernel_size_samples = train_dict['kernel_size_samples']
    n_filters_samples = train_dict['n_filters_samples']
    lr_samples = train_dict['lr_samples']
    window_size_samples = train_dict['window_size_samples']
    overlap_prior_samples = train_dict['overlap_prior_samples']
    batch_size_samples = train_dict['batch_size_samples']
    maxpool_samples = train_dict['maxpool_samples']
    telescopic_prior_samples = train_dict['telescopic_prior_samples']

    if type(segy_filename) is list:
        segy_filename = segy_filename[0]

    # Make a master segy object
    segy_obj = segy_read(segy_file=segy_filename,
                           mode='create',
                           read_direc='full',
                           inp_res=inp_res)

    # Define how many segy-cubes we're dealing with
    segy_obj.cube_num = 1
    segy_obj.data = np.expand_dims(segy_obj.data, axis=len(segy_obj.data.shape))

    train_points = convert(train_files, facies_names)
    seis_data, masks, labels = points_to_images(train_points, segy_obj)

    test_points = convert(test_files, facies_names)
    test_seis_data, test_masks, test_labels = points_to_images(test_points, segy_obj)

    n_classes = len(facies_names)

    if not os.path.exists(save_location_global):
        os.makedirs(save_location_global)

    acc_file = open(save_location_global + 'hyper_tuning.txt', 'w+')
    runtime_file = open(save_location_global + 'runtimes.txt', 'w+')
    runtime_file.write(f" \ttrain_time\tprediction_time\n")

    x_shape = test_labels.shape[1]
    y_shape = test_labels.shape[2]

    for i in range(len(num_conv_layers_samples)):

        if maxpool_samples[i]:
            for k in range(num_conv_layers_samples[i] // 2):
                x_shape = np.ceil(x_shape / 2)
                y_shape = np.ceil(y_shape / 2)

            for k in range(num_conv_layers_samples[i] // 2):
                x_shape = int(x_shape * 2)
                y_shape = int(y_shape * 2)

            test_seis_data = np.pad(test_seis_data, ((0, 0), (0, x_shape - test_seis_data.shape[1]), (0, y_shape - test_seis_data.shape[2]),
                                       (0, 0)), mode='constant', constant_values=0)
            test_labels = np.pad(test_labels, ((0, 0), (0, x_shape - test_labels.shape[1]), (0, y_shape - test_labels.shape[2]),
                                       ), mode='constant', constant_values=0)
            test_masks = np.pad(test_masks, ((0, 0), (0, x_shape - test_masks.shape[1]), (0, y_shape - test_masks.shape[2]),
                                       (0, 0)), mode='constant', constant_values=0)

        model = build_model(seis_data, num_conv_layers_samples[i], num_dil_layers_samples[i], n_filters_samples[i],
                            int(kernel_size_samples[i]), n_classes, maxpool=maxpool_samples[i],
                            telescopic=telescopic_prior_samples[i])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_samples[i]),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        window_step = int(np.ceil(window_size_samples[i] * (1 - overlap_prior_samples[i] / 100)))

        seis_train, mask_train, labels_train = patches_creator(seis_data, masks, labels, window_size_samples[i], window_step)
        seis_train = seis_train
        mask_train = mask_train
        labels_train = labels_train

        start = time.time()

        history = LossHistory()
        model.fit([seis_train, mask_train], labels_train, batch_size=batch_size_samples[i], epochs=n_epoch, verbose=1,
                  callbacks=[history], validation_split=validation_split)

        train_time = time.time() - start

        save_location = save_location_global + f'set_{i}/'
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        model.save(save_location + 'trained.h5')

        binary_mask_predict = np.ones(test_seis_data.shape)
        start = time.time()
        prediction_prob = model.predict([test_seis_data, binary_mask_predict])
        prediction_time = time.time() - start

        for k in range(prediction_prob.shape[-1]):
            plt.imsave(save_location + f'prob_{k}.jpeg', np.transpose(prediction_prob[0, ..., k]), vmin=0, vmax=1)

        prediction = np.argmax(prediction_prob, axis=-1)
        plt.imsave(save_location + f'pred.jpeg', np.transpose(prediction[0]))
        runtime_file.write(f"{i}\t{train_time}\t{prediction_time}\n")

        labeled_rows, labeled_cols = np.where(test_masks[0, ..., 0] == 1)
        true_test_labels = test_labels[0, labeled_rows, labeled_cols]

        pred_test_labels = prediction[0, labeled_rows, labeled_cols]
        test_acc = np.sum(pred_test_labels == true_test_labels) / len(true_test_labels)
        printout(history, save_location, true_test_labels, pred_test_labels, facies_names)

        acc_file.write(f"set_{i}:\twindow size: {window_size_samples[i]}\tnum_conv_layers: {num_conv_layers_samples[i]}\t"
                    f"num_dil_layers: {num_dil_layers_samples[i]}\tn_filters: {n_filters_samples[i]}\t"
                    f"kernel_size: {kernel_size_samples[i]}\tlr: {lr_samples[i]}\toverlap: {overlap_prior_samples[i]}\t"
                    f"batch size: {batch_size_samples[i]}\tmaxpool: {maxpool_samples[i]}\t"
                    f"telescopic: {telescopic_prior_samples[i]}\taccuracy: {test_acc}\t\n\n")

    acc_file.close()
    runtime_file.close()

