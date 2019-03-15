import random
import numpy as np

def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

    # Apply some random transformations
    random_transformation = np.random.randint(1,8)
    if random_transformation == 1:  # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        pass

    return patch_img, patch_mask

#n_patches : 패치의 갯수
#이미지에서 무작위로 sz x sz 크기의 영역을 잘라낸 후, 적절한 변환을 거쳐 패치를 생성한다.
#생성된 패치들은 훈련, 검증에 사용된다.
#패치의 크기가 작을수록 적은 영역만 이용하는 것이다.
#패치의 갯수가 많을수록 많은 양의 자료를 이용하는 것이다.
#패치를 생성할 때 임의의 변환을 시도한다. 따라서 같은 영상의 패치이더라도 변환이 어떻게 되는가에 따라 달라질 수 있다.
#이는 모델의 강건성을 높이기 위함이다.
#위성영상 1개당 패치는 921개가 적당하며, 훈련셋 패치는 691개, 검증셋 패치는 230개이다.
#이는 영상의 변환이 없는 상황에서이며, 변환은 총 6가지로 원본인 경우를 포함하여 7가지의 영상을 반환한다.
#이를 고려하면 위성영상 1개당 전체 패치는 약 6300개, 훈련셋 패치는 4800개, 검증셋 패치는 1500개이다.

#DG영상 1개당 패치는 41개가 적당하며, 훈련셋 패치는 30개, 검증셋 패치는 11개이다.
# 변환을 고려하면 DG영상 1개당 전체 패치는 25만개, 훈련셋 패치는 19만개, 검증셋 패치는 6만개이다.
def get_patches(x_dict, y_dict, n_patches, sz=160):
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)


