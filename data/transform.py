import cv2
import numpy as np
import types
from numpy import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.k = 0
    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        # import pdb; pdb.set_trace()
        # cv2.imwrite('test/'+str(self.k)+'.jpg', img)
        self.k += 1
        # print(self.k)
        return img, boxes, labels
    
class random_photo(object):
    def __call__(self, image, boxes=None, labels=None):
        assert image.dtype == np.uint8
        if boxes is not None:
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=boxes[i][0], y1=boxes[i][1], x2=boxes[i][2], y2=boxes[i][3], label = labels[i]) for i in range(len(boxes))
            ], shape = image.shape)
            seq = iaa.Sequential([
                    iaa.SomeOf(2, [
                        iaa.GaussianBlur(sigma=(0.5, 1)),
                        # iaa.MedianBlur(k=(3, 5)),
                        iaa.BilateralBlur(d=(3,5), sigma_color=(10, 250), sigma_space=(10, 250)),
                        iaa.MotionBlur(k=5, angle=(0, 360), direction=(-0.5, 0.5))
                    ], random_order = True),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.Sometimes(0.5, iaa.Add((0, 40))),
                    iaa.Sequential([
                        iaa.SomeOf(2, [
                            iaa.LinearContrast((0.75, 1.5)),
                            iaa.GammaContrast((0.5, 1.5)),
                            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                        ], random_order=True),
                        iaa.SomeOf(1, [
                            iaa.MultiplyHueAndSaturation((0.5, 1.2)),
                            iaa.MultiplyHue((0.5, 1.2)),
                            iaa.MultiplySaturation((0.5, 1.2))
                        ], random_order=True),
                    ]), 
            ])
        
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        # import pdb; pdb.set_trace()
        bbs = np.zeros((len(bbs_aug), 4))
        labels = np.zeros((len(bbs_aug), 1))
        for k in range(len(bbs_aug)):
            t = bbs_aug[k]
            bbs[k] += (t.x1, t.y1, t.x2, t.y2)
            labels[k] += t.label
        # bbs = np.array([t.x1, t.y1, t.x1, t.y2] for t in bbs_aug)
        
        return image_aug, bbs, labels
 
class lib_augment(object):
    def __init__(self, size):
        self.size = size
    #     if boxes is not None:
    #         bbs = BoundingBoxesOnImage([
    #             BoundingBox(x1=x, y1=y, x2=xx, y2=yy) for (x, y, xx, yy) in boxes
    #         ], shape = img.shape)
    def __call__(self, image, boxes=None, labels=None):
        raw_bbs = boxes
        raw_labels = labels
        
        if boxes is not None:
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=boxes[i][0], y1=boxes[i][1], x2=boxes[i][2], y2=boxes[i][3], label = labels[i]) for i in range(len(boxes))
            ], shape = image.shape)
            seq = iaa.Sequential([
                    iaa.SomeOf(1, [
                        iaa.GaussianBlur(sigma=(0.5, 1)),
                        # iaa.MedianBlur(k=(3, 5)),
                        iaa.BilateralBlur(d=(3,5), sigma_color=(10, 250), sigma_space=(10, 250)),
                        iaa.MotionBlur(k=5, angle=(0, 360), direction=(-0.5, 0.5))
                    ]),
                    # iaa.Fliplr(0.5)
                    # iaa.LinearContrast((0.75, 1.5)),
                    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
                    # iaa.PerspectiveTransform(scale=(0.01, 0.05), keep_size=True, fit_output=True, cval=(0)),
                    # iaa.SomeOf(1, [
                    #     iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=[0, 1], cval=0, mode='constant'),
                    #     iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, order=[0, 1], cval=0, mode='constant'),
                    #     iaa.Affine(rotate = (-7, 7), order=[0, 1], cval=0, mode='constant'),
                    #     iaa.Affine(shear = (-7, 7), order=[0, 1], cval=0, mode='constant'),
                    # ]),
                    
                    # # #Photometric distortion
                    # iaa.Sometimes(0.5, iaa.Add((0, 40))),
                    # iaa.Sequential([
                    #     iaa.GammaContrast((0.5, 1.5)),
                    #     iaa.SomeOf(1, [
                    #         iaa.MultiplyHueAndSaturation((0.5, 1.2)),
                    #         iaa.MultiplyHue((0.5, 1.2)),
                    #         iaa.MultiplySaturation((0.5, 1.2))
                    #     ], random_order=True),
                    #     iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                    # ]), 
            ])
        
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        # import pdb; pdb.set_trace()
        bbs = np.zeros((len(bbs_aug), 4))
        labels = np.zeros((len(bbs_aug), 1))
        for k in range(len(bbs_aug)):
            t = bbs_aug[k]
            bbs[k] += (t.x1, t.y1, t.x2, t.y2)
            labels[k] += t.label
        # bbs = np.array([t.x1, t.y1, t.x1, t.y2] for t in bbs_aug)
        
        return image_aug, bbs, labels
        
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class Resize(object):
    def __init__(self, size=640):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels


class Augmentation(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ToAbsoluteCoords(),            # 将归一化的相对坐标转换为绝对坐标
            random_photo(),            
            ConvertFromInts(),             # 将int类型转换为float32类型
            lib_augment(self.size),
            # PhotometricDistort(),          # 图像颜色增强
            Expand(self.mean),             # 扩充增强
            RandomSampleCrop(),            # 随机剪裁
            RandomMirror(),                # 随机水平镜像
            ToPercentCoords(),             # 将绝对坐标转换为归一化的相对坐标
            Resize(self.size),             # resize操作
            Normalize(self.mean, self.std) # 图像颜色归一化
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        # resize
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        # normalize
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels