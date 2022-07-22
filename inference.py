import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from tqdm import tqdm
import skimage
from skimage.measure import label

# путь до тестовых данных
x_test_dir = f'data'


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background', 'sub_track', 'track', 'train']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask[mask == 6] = 1
        mask[mask == 7] = 2
        mask[mask == 10] = 3
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),


        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    
    test_transform = [
        albu.Resize(height=1024, width=1024),
        albu.CLAHE(p=1),
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocessing_fn_resnext = smp.encoders.get_preprocessing_fn('se_resnext50_32x4d', 'imagenet')
preprocessing_fn_resnet = smp.encoders.get_preprocessing_fn('resnet18', 'imagenet')
preprocessing_fn_effnet = smp.encoders.get_preprocessing_fn('efficientnet-b2', 'imagenet')

model_se_resnet = torch.load('models/se_resnext50_32x4d.pth')
model_b2 = torch.load('models/efficient-b2.pth')
model_resnet18 = torch.load('models/resnet18.pth')

albus = get_validation_augmentation()

def get_image(image_path: str, image_size: tuple, preprocessing_fn):
    image = cv2.imread(image_path)
    img_shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = preprocessing_fn(image)
    image = to_tensor(image)

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    return x_tensor, img_shape

def get_model_predicts(model, x_tensor):

    predicted_mask = model.predict(x_tensor)
    predicted_mask = (predicted_mask.squeeze().cpu().numpy().round())

    predicted_mask = predicted_mask.transpose(1, 2, 0)

    return predicted_mask


def get_ensemble_predicts(first_predict, second_predict, third_predict, treshold):

    channel1 = (first_predict[:, :, 0] + second_predict[:, :, 0] + third_predict[:, :, 0]) / 3
    channel1[channel1 > treshold] = 6
    channel1[channel1 < treshold] = 0

    channel2 = (first_predict[:, :, 1] + second_predict[:, :, 1] + third_predict[:, :, 1]) / 3
    channel2[channel2 > treshold] = 7
    channel2[channel2 < treshold] = 0

    channel3 = (first_predict[:, :, 2] + second_predict[:, :, 2] + third_predict[:, :, 2]) / 3
    channel3[channel3 > treshold] = 10
    channel3[channel3 < treshold] = 0

    return channel1 + channel2 + channel3


def post_processing_mask(predicted_mask, image_shape):
    if len(predicted_mask.shape) > 2:
        predicted_mask[:, :, 0][predicted_mask[:, :, 0] == 1] = 6
        predicted_mask[:, :, 1][predicted_mask[:, :, 1] == 1] = 7
        predicted_mask[:, :, 2][predicted_mask[:, :, 2] == 1] = 10

        predicted_mask = predicted_mask[:, :, 0] + predicted_mask[:, :, 1] + predicted_mask[:, :, 2]

    if 13 in np.unique(predicted_mask):
        predicted_mask = np.where(predicted_mask == 13, 7, predicted_mask)
        print('_____Наложили основной трек на саб трек______')

    if 16 in np.unique(predicted_mask):
        predicted_mask = np.where(predicted_mask == 16, 10, predicted_mask)
        print('_____Наложили поезд на саб трек______')

    if 17 in np.unique(predicted_mask):
        predicted_mask = np.where(predicted_mask == 17, 10, predicted_mask)
        print('_____Наложили поезд на основной трек______')

    if 23 in np.unique(predicted_mask):
        predicted_mask = np.where(predicted_mask == 23, 10, predicted_mask)
        print('_____Наложили поезд на треки______')

    resized_mask = skimage.transform.resize(predicted_mask,
                               image_shape,
                               mode='edge',
                               anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               preserve_range=True,
                               order=0)

    return resized_mask




for img_path in tqdm(os.listdir(x_test_dir)):
    x_tensor, image_shape = get_image(f'{x_test_dir}/{img_path}', (1024, 1024), preprocessing_fn_resnext)
    model_se_resnext_predict = get_model_predicts(model_se_resnet, x_tensor)

    x_tensor, image_shape = get_image(f'{x_test_dir}/{img_path}', (1024, 1024), preprocessing_fn_effnet)
    model_b2_predict = get_model_predicts(model_b2, x_tensor)

    x_tensor, image_shape = get_image(f'{x_test_dir}/{img_path}', (1024, 1024), preprocessing_fn_resnet)
    model_resnet18_predict = get_model_predicts(model_resnet18, x_tensor)

    predicted_mask = get_ensemble_predicts(model_se_resnext_predict, model_b2_predict, model_resnet18_predict, 0.5)

    resized_mask = post_processing_mask(predicted_mask, image_shape)

    cv2.imwrite(f'results/{img_path}', resized_mask)

    # break