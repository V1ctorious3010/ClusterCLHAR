import numpy as np
import tensorflow as tf
from sklearn.cluster import Birch,KMeans
from sklearnex import patch_sklearn
from Augment import resample_random
from module import contrastive_loss

import numpy as np
import tensorflow as tf
from sklearn.cluster import Birch,KMeans
from sklearnex import patch_sklearn
from Augment import resample_random
from module import contrastive_loss
from tensorflow.keras.utils import to_categorical
def get_data(data_name, uci_test_group=None):
    NPY_PATH = "/kaggle/working/"
    x_data = np.load(NPY_PATH + 'UCI_X.npy')     # (10299, 128, 9)
    y_data = np.load(NPY_PATH + 'UCI_y.npy')     # (10299, 6)
    subjects = np.load(NPY_PATH + 'UCI_sub.npy') # (10299,)
    y_data = y_data - 1

    test_groups = {
        1: (1, 6),  
        2: (7, 12),
        3: (13, 18),
        4: (19, 24),
        5: (25, 30)
    }

    min_sub, max_sub = test_groups[uci_test_group]
    test_mask = (subjects >= min_sub) & (subjects <= max_sub)
    train_mask = ~test_mask
    
    x_train, y_train = x_data[train_mask], y_data[train_mask]
    x_test, y_test = x_data[test_mask], y_data[test_mask]

    np.random.seed(888)
    p_train = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p_train], y_train[p_train]
    y_train = to_categorical(y_train, num_classes=6)
    y_test = to_categorical(y_test, num_classes=6)
    return x_train, y_train, x_test, y_test


def get_cluster(cluster_name,cluster_num):
    if cluster_name == 'birch':
        cluster = Birch(threshold=0.1, n_clusters=cluster_num)
    elif cluster_name == 'kmeans':
        cluster = KMeans(n_clusters=cluster_num)
    return cluster

def train_step(xis, xjs, model, optimizer,cluster,args):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)
        loss = contrastive_loss(zis,zjs,cluster,args)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(loss)

def train(model,x_data,args):
    optimizer = tf.keras.optimizers.Adam(args.lr)
    epochs = args.epoch
    batch_size = args.batch_size

    patch_sklearn()
    cluster = get_cluster(args.cluster,args.cluster_num)

    cur_loss = 1e9
    seed = len(x_data)
    for epoch in range(epochs):
        loss_epoch = []
        train_loss_dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(seed,reshuffle_each_iteration=True).batch(batch_size)
        for x in train_loss_dataset:
            xis = resample_random(x)
            xjs = x
            loss = train_step(xis, xjs, model, optimizer,cluster,args)
            loss_epoch.append(loss)
        print("epoch{}===>loss:{}".format(epoch + 1, np.mean(loss_epoch)))
        if epoch > epochs//2 and np.mean(loss_epoch) < cur_loss:
            tf.keras.models.save_model(model,'contrastive_model/'+'{}_cluster_{}_batchsize_{}_epoch_{}'.format(args.dataset,args.cluster,args.batch_size,args.epoch))
            cur_loss = np.mean(loss_epoch)
