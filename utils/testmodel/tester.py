import itertools
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# 使用GPU进行计算的GPU初始化代码；如果不使用则会出现cudNN无法使用的报错

def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return 0


def create_test_image_data(DATADIR, IMG_SIZE):
    test_image_data = []
    CATEGORIES = os.listdir(DATADIR)

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        if 'NoFire' in category:
            class_num = 1
        elif 'Fire' in category:
            class_num = 0

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_image_data.append([new_array, class_num])  # add this to our test_image_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    return test_image_data


def random_shuffle_data(test_image_data):
    print(len(test_image_data))
    random.shuffle(test_image_data)
    return test_image_data


def create_test_labels(test_image_data, test_image_num=871):
    c = 0
    test_labels = np.zeros((test_image_num, 1))
    for sample in test_image_data:
        test_labels[c] = (sample[1])
        c += 1
    print(c)
    actual_labels = (test_labels.reshape(test_image_num, ))
    print(actual_labels.shape)
    actual_labels.astype(int)
    return actual_labels


def create_dataset(test_image_data, IMG_SIZE=64):
    X = []
    Y = []

    for features, label in test_image_data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X = X / 255.0
    # X.shape[1:]

    Y = np.array(Y)
    return X, Y


def load_tf_h5_model(model_path='my_model.h5'):
    model = tf.keras.models.load_model(model_path)
    return model


def predicte_labels(X, model, test_image_num):
    predicted_labels = model.predict_classes(X)
    predicted_labels = (predicted_labels.reshape(test_image_num, ))
    predicted_labels.astype(int)

    return predicted_labels


def begain_compute(actual_labels, predicted_labels):
    cm = confusion_matrix(actual_labels, predicted_labels)
    # test_batches.class_indices
    cm_plot_labels = ['Fire', 'No Fire']

    # 真正
    tp = cm[0][0]
    # 假负
    fn = cm[0][1]
    # 假正
    fp = cm[1][0]
    # 真负
    tn = cm[1][1]

    print("tp" + ' ' + str(tp))
    print("fn" + ' ' + str(fn))
    print("fp" + ' ' + str(fp))
    print("tn" + ' ' + str(tn))
    # 召回率
    Recall = tp / (tp + fn)
    # 准确率
    Precision = tp / (tp + fp)
    f_measure = 2 * ((Precision * Recall) / (Precision + Recall))

    print('Precision=', Precision, 'Recall=', Recall, 'f_measure=', f_measure)

    return cm, cm_plot_labels, Precision, Recall, f_measure


def dispaly_model_summary(model):
    # 显示模型的结构
    model.summary()
    return 0


def evaluate_model(X, Y, model):
    result = model.evaluate(X, Y)

    return result


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues',
                          figure_save_path='test.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig(figure_save_path)
    plt.close(figure_save_path)

    return 0


def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        # directory already exists
        pass

    return 0


if __name__ == '__main__':
    init_gpu()
    test_image_data = create_test_image_data(DATADIR='Datasets/Test_Dataset1__Our_Own_Dataset', IMG_SIZE=64)
    shuffled_test_image_data = random_shuffle_data(test_image_data)
    test_image_num = len(shuffled_test_image_data)
    actual_labels = create_test_labels(shuffled_test_image_data, test_image_num=test_image_num)
    X, Y = create_dataset(test_image_data=shuffled_test_image_data, IMG_SIZE=64)
    disk_root = '/run/media/expoli/ed5c6898-13d8-4c60-a590-6493de9822bc/tensorflow/FireNet-LightWeight-Network-for-Fire-Detection/'
    project_root = 'Codes/python-reformat/'
    model_root = 'result4/results00/'
    cp_path = 'checkpoint_path/'
    model_path = disk_root + project_root + model_root + cp_path
    models_path = os.listdir(model_path)

    Precisions = []
    recalls = []
    f_measures = []
    for path in models_path:
        fileObject = open(disk_root + project_root + model_root + 'recalls-0.txt', 'a')
        model = load_tf_h5_model(
            model_path=model_path + path)
        # dispaly_model_summary(model)
        predicted_labels = predicte_labels(X=X, model=model, test_image_num=test_image_num)
        cm, cm_plot_labels, Precision, Recall, f_measure = begain_compute(actual_labels=actual_labels,
                                                                          predicted_labels=predicted_labels)
        fileObject.write(path + path + ':')
        fileObject.write('\t')
        fileObject.write(str(Precision))
        fileObject.write('\t')
        fileObject.write(str(Recall))
        fileObject.write('\t')
        fileObject.write(str(f_measure))
        fileObject.write('\t')
        # plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix', figure_save_path='result/train05')
        fileObject.write(str(evaluate_model(X, Y, model)))
        fileObject.write('\n')
        fileObject.close()

    # fileObject = open('recalls.txt', 'w')
    # for i,precision,recall,f_measure in Precisions,recalls,f_measures:
    #     fileObject.write(i+':')
    #     fileObject.write(precision)
    #     fileObject.writable()
    #     fileObject.write(recall)
    #     fileObject.writable()
    #     fileObject.write(f_measure)
    #     fileObject.write('\n')
