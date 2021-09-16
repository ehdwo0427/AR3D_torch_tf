import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf



from tqdm import tqdm

os.sys.path.append('/home/subin/AR3D/AR3D/')
from config_tf import config
from dataloader_tf import load_data, DataLoader, load_data_2
from network import AR3D_tf as AR3D


tf.debugging.set_log_device_placement(True)

def train_model():
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    
    x_train, y_train = load_data(config.dataset, "train")
    x_valid, y_valid = load_data(config.dataset, "val")

    train_loader = DataLoader(x_train, y_train, config.batch_size, 'train', True)
    test_loader =  DataLoader(x_train, y_train, config.batch_size, 'test', True)

    if config.model_name == 'AR3D':
        
        ## multi gpu test1 
        stratege = tf.distribute.MirroredStrategy()

        with stratege.scope():
        # with tf.device('/device:GPU:0'):
            model = AR3D.AR3D(num_classes = config.num_classes, AR3D_V = config.ar3d_version, SFE_type = config.sfe_type, 
                              attention_method = config.attention, reduction_ratio = config.reduction_ratio, hidden_unit = config.hidden_units)
        
        model.build(input_shape=(config.batch_size, config.frames_per_clips, 112, 112, 3))
        opt = tf.keras.optimizers.Adam(learning_rate=config.lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


        
        ## multi gpu test2

        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     for gpu in gpus:
        #         with tf.device(gpu.name):
        #             pass

        
    
def train_model_test():
    x_train, y_train = load_data(config.dataset, "train")
    x_valid, y_valid = load_data(config.dataset, "val")
    ##    LDJ    ##
    # ALL_VIDEO_LIST = np.loadtxt('/Users/dj/Downloads/Handover/labels/train_list_1234.txt', delimiter=' ', dtype='str')
    # LABEL_LIST = ALL_VIDEO_LIST[:, [-1]]
    # label = np.array(LABEL_LIST, dtype=np.int) - 1
    # label = np.delete(label, [173])
    #####
    # print(x_train)
    # train_loader = DataLoader(x_train, label, config.batch_size, 'train', shuffle=True)
    train_loader = DataLoader(x_train, y_train, config.batch_size, 'train', shuffle=True)
    test_loader =  DataLoader(x_valid, y_valid, config.batch_size, 'test', shuffle=True)

    model = AR3D.AR3D(num_classes = config.num_classes, AR3D_V = config.ar3d_version, SFE_type = config.sfe_type, 
                    attention_method = config.attention, reduction_ratio = config.reduction_ratio, hidden_unit = config.hidden_units)

    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Accuracy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.Accuracy()

    epochs = 50
    for epoch_index in range(epochs):
        for x_train_batch, y_train_batch in tqdm(train_loader):
            with tf.GradientTape() as tape:
                predictions = model(x_train_batch, training=True)
                loss_value = loss(y_train_batch, predictions)
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss_value)
            train_accuracy.update_state(tf.argmax(y_train_batch, axis=1), tf.argmax(predictions, axis=1))

        for x_test_batch, y_test_batch in tqdm(test_loader):
            predictions = model(x_test_batch)
            loss_value = loss(y_test_batch, predictions)
            test_loss.update_state(loss_value)
            test_accuracy.update_state(tf.argmax(y_test_batch, axis=1), tf.argmax(predictions, axis=1))

        print('epoch: {}/{}, train loss: {:.4f}, train accuracy: {:.4f}, test loss: {:.4f}, test accuracy: {:.4f}'.format(
            epoch_index + 1, epochs, train_loss.result().numpy(), train_accuracy.result().numpy(), test_loss.result().numpy(), test_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        

             
if __name__ == "__main__":
    train_model_test()
