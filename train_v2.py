from gen_patches import *

import numpy as np

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
import tensorflow as tf
from keras import backend as K

import os.path
import tifffile as tiff
import cv2
import sys





N_EPOCHS = 50
PATCH_SZ = 160   # 패치의 크기, 이미지 크기 * 0.25를 넘으면 안됨(DEEPGLOBE 영상은 1024 크기이므로 256의 크기를 넘으면 안됨), 32의 배수이어야 함, 너무 작으면 업샘플링에 어려움이 있음
BATCH_SIZE = 32
TRAIN_SZ = 3000  # 훈련 자료 패치의 갯수
VAL_SZ = 1000    # 검증 자료 패치의 갯수, 훈련-검증 비율을 3:1로 잡았으므로 패치의 갯수도 4:1로 계산하는 것이 적절할 듯.


img_size_ori = 101

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv2D(dim, 3, activation=acti, padding='same')(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, padding='same')(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)




# flag = 'd' or ''n'
flag = 'd'
#flag = sys.argv[1]

weights_path = './weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

if flag == 'd':
    weights_path += '/unet_weights_dg.hdf5'
else:
    weights_path += '/unet_weights_ngii.hdf5'

# 입력 이미지 경로 및 파일명 정보
# file_list_image내의 파일은 숫자로만 되어 있음
# DEEPGLOBE 자료는 이미지 JPG, GT PNG
# 국토정보지리원 자료는 이미지 TIF, GT TIF
if flag == 'd':
    path_dir_image = './data/dg/image'
    path_dir_gt = './data/dg/gt'
else:
    path_dir_image = './data/ngii/image'
    path_dir_gt = './data/ngii/gt'

file_list_image = os.listdir(path_dir_image)
file_list_gt = os.listdir(path_dir_gt)
file_list_image.sort()
file_list_gt.sort()

X_DICT_TRAIN = dict()
Y_DICT_TRAIN = dict()
X_DICT_VALIDATION = dict()
Y_DICT_VALIDATION = dict()

t_start = cv2.getTickCount()
t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
print("[*] 이미지 로드 시작 Time: %.3f ms" % t_total)
cnt = 1
# DEEPGLOBE 이미지 불러오기
for img_id in file_list_image[0:100]:
    filenum = img_id.split('.')[0] # 파일 번호 저장
    ext = img_id.split('.')[1]
    path_img = path_dir_image + '/' + img_id # 이미지 파일 경로 저장
    if(ext != 'db'):
        # GT 파일 경로 저장
        if(flag == 'd'):
            path_gt = path_dir_gt + '/' + filenum + '.png'
            img_m = normalize(cv2.imread(path_img, cv2.IMREAD_COLOR))
            mask_img = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
            mask = mask_img.reshape((mask_img.shape[0], mask_img.shape[1], 1)) / 255
        else:
            path_gt = path_dir_gt + '/' + filenum + '_GT.tif'
            img_m = normalize(tiff.imread(path_img))
            mask_img = tiff.imread(path_gt)
            mask = mask_img.reshape((mask_img.shape[0], mask_img.shape[1], 1))

        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[filenum] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[filenum] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[filenum] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[filenum] = mask[train_xsz:, :, :]

        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
        print('(' + str(cnt) + '/' + str(file_list_image.__len__()) + ') ' + filenum + ' read / 시간: %.3f' % t_total)
        cnt += 1

t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
print("[*] 이미지 로드 완료 Time: %.3f ms" % t_total)
print("[*] 학습 시작 Time: %.3f ms" % t_total)
x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
x_valid, y_valid = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)

# U-Net의 기본 모델은 입력 채널 64, 깊이 4인 경우임
# 깊이가 4이면 입력 패치 크기는 16의 배수이어야 함
# 깊이가 5이면 입력 패치 크기는 32의 배수이어야 함
model = UNet((PATCH_SZ,PATCH_SZ,3),start_ch=64,depth=4,batchnorm=True)
#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[mean_iou])
model.summary()

early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("./keras_dg_v3.model", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr],shuffle=True)

model = load_model("./keras_dg_v3.model")

preds_valid = model.predict(x_valid)

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

print('IoU Best: ' + str(iou_best))
print('Threshold: ' + str(threshold_best))
#plt.plot(thresholds, ious)
#plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
#plt.xlabel("Threshold")
#plt.ylabel("IoU")
#plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
#plt.legend()