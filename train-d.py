from numpy.core.multiarray import ndarray
from unet_model import *
from gen_patches import *

import os.path
import tifffile as tiff
import cv2
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

N_BANDS = 3 # RGB
N_CLASSES = 1  # 도로
CLASS_WEIGHTS = [1]
N_EPOCHS = 100
PATCH_SZ = 160   # 패치의 크기, 이미지 크기 * 0.25를 넘으면 안됨(DEEPGLOBE 영상은 1024 크기이므로 256의 크기를 넘으면 안됨), 32의 배수이어야 함, 너무 작으면 업샘플링에 어려움이 있음
BATCH_SIZE = 64
TRAIN_SZ = 900  # 훈련 자료 패치의 갯수
VAL_SZ = 300    # 검증 자료 패치의 갯수, 훈련-검증 비율을 3:1로 잡았으므로 패치의 갯수도 4:1로 계산하는 것이 적절할 듯.

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, class_weights=CLASS_WEIGHTS)

# flag = 'd' or ''n'
flag = sys.argv[1]

weights_path = 'weights'
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

if __name__ == '__main__':
    t_start = cv2.getTickCount()

    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
    print("[*] 이미지 로드 시작 Time: %.3f ms" % t_total)
    cnt = 1
    # DEEPGLOBE 이미지 불러오기
    for img_id in file_list_image[1:10]:
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

    def train_net():
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
        print("[*] 학습 시작 Time: %.3f ms" % t_total)
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)

        if flag=='d':
            csv_logger = CSVLogger('log_unet_dg.csv', append=True, separator=';')
        else:
            csv_logger = CSVLogger('log_unet_ngii.csv', append=True, separator=';')

        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
        print("[*] 모델 설정 완료 Time: %.3f ms" % t_total)

        model.summary()

        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=1, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))

        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
        print("[*] 모델 학습 완료 Time: %.3f ms" % t_total)

        results = model.evaluate(x_val, y_val)
        print('Test accuracy: ', results[1])

        return model

    train_net()