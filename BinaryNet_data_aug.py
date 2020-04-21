import argparse
import os

import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 保存文件路径配置
from projectconfig import pathconfig
# 图像预处理
from pyimagesearch.datasets import SimpleDatasetLoader as SDL
from pyimagesearch.nn.conv import BinaryNet as BinNet
from pyimagesearch.preprocessing import AspectAwarePreprocessor as AAP
from pyimagesearch.preprocessing import ImageToArrayPreprocessor as IAP
# GPU 初始化
from tfsettings.gpu import InitGpu
# 回调函数
from tfsettings.traincallbacks import modelcheckpoint as TF_CB_chcekpoint
from tfsettings.traincallbacks import tensorboard as TF_CB_tensorboard
# 绘制训练结果
from utils.trainplot import plot as my_plot_tool

ap = argparse.ArgumentParser()  # 从命令中读取参数
ap.add_argument('-d', '--train_dataset', required=True, help='path to train dataset')
ap.add_argument('-e', '--epochs', required=True, help='training epochs number')
ap.add_argument('-t', '--test_dataset', required=True, help='path to test dataset')
args = vars(ap.parse_args())

print('[INFO] initing gpu.....')
gpu = InitGpu.InitGpu()
gpu.init()

print("[INFO] loading train images...")  # 预处理图片
imagePaths = [x for x in list(paths.list_images(args['train_dataset'])) if x.split(os.path.sep)[-2] != 'jpg']
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AAP.AspectAwarePreprocesser(64, 64)  # 重设大小
iap = IAP.ImageToArrayPreprocess()  # 转换成张量

sdl = SDL.SimpleDatasetLoader(preprocessors=[aap, iap])
(train_data, train_labels) = sdl.load(imagePaths, verbose=500)
train_data = train_data.astype('float') / 255.0

(trainX, validX, trainY, validY) = train_test_split(train_data, train_labels, test_size=0.30, random_state=43)

trainY = LabelBinarizer().fit_transform(trainY)
validY = LabelBinarizer().fit_transform(validY)

# construct the image generator for data augmentation 数据增强
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
# opt = SGD(lr=0.05)
opt = 'adam'
batch_size = 128
model = BinNet.BinaryNet.build(width=64, height=64, depth=3)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
epochs = int(args['epochs'])
config_path = pathconfig.PathConfig()  # 实例化对象
config_path.set_root_path(new_root_path='results/results03')  # 设置训练结果保存路径

# 构建回调函数
# 生成 tersorborad log 文件
cb_tensorboard = TF_CB_tensorboard.TensorBoardCallBack()
cb_tensorboard.set_log_path(config_path.get_tensorboard_path())
# 保存 cp 文件
cb_chcekpoint = TF_CB_chcekpoint.ModelCheckpointCallBack()
cb_chcekpoint.set_checkpoint_path(config_path.get_checkpoint_path())

H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(validX, validY), steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1,
              callbacks=[cb_tensorboard.build_cb(), cb_chcekpoint.build_cb()])
# 保存模型
model.save(filepath=config_path.get_model_save_path())

# evaluate the network
from utils.testplot.plot import ConfusionMatrix

confusion_matrix = ConfusionMatrix()
confusion_matrix = confusion_matrix.begin(args['test_dataset'], model=model, batch_size=batch_size)

# 绘制结果
my_plot_tool.plot_train_loss_and_acc(epochs=epochs, H=H)
