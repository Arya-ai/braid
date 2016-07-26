from __future__ import absolute_import
import numpy as np
import cv2
import warnings
from .preprocessing import (create_fixed_image_shape, one_hot)


class Transformer(object):
    '''Class for performing augmentation on the training/validation dataset.
    Augmentations:
        + Random crops
        + Horizontal/vertical random flips
        + Random zooming
    '''

    def __init__(self, num_classes=None, reshape=(), random_crop=(),
                 hflip=False, vflip=False, zoom=[], seed=1234):
        assert (isinstance(reshape, tuple) and isinstance(
            random_crop, tuple) and isinstance(zoom, list))
        if len(reshape) == 2 or (len(reshape) > 2 and reshape[2] == 1):
            self.reshape = tuple(reshape[:2])
        else:
            self.reshape = tuple(reshape[:3])
        self.random_crop = random_crop[:2] if len(
            random_crop) in {2, 3} else False
        self.out_shape = self.random_crop if self.random_crop else self.reshape
        if len(self.reshape) == 3 and len(self.out_shape) == 2:
            self.out_shape = tuple(list(self.out_shape) + [self.reshape[2]])
        self.zoom = zoom if len(zoom) == 2 else False
        self.hflip = hflip
        self.vflip = vflip
        self.rng = np.random.RandomState(seed=seed)
        self.num_classes = num_classes
        if self.num_classes is None:
            warnings.warn(
                "[WARN] Transformer: num_classes not provided to"
                " `Transformer` class;"
                " the label will not be converted to one hot vector.")
        if self.zoom:
            assert (self.zoom[0] <= 1. and self.zoom[0] > 0. and
                    self.zoom[1] <= 2. and self.zoom[1] >= 1.), (
                "[ERROR] Transformer: `zoom` takes a list [a, b];"
                " where 0 < a <= 1 and 1 <= b <= 2.")
        assert self.hflip in {True, False}, (
            "[ERROR] Transformer: `hflip` should be a bool")
        assert self.vflip in {True, False}, (
            "[ERROR] Transformer: `vflip` should be a bool")
        self._print_params()

    def _print_params(self):
        print "[INFO] Transformer: Transformer params:"
        print "\treshape: {}".format(self.reshape)
        print "\trandom_crop: {}".format(self.random_crop)
        print "\tzoom: {}".format(self.zoom)
        print "\thflip: {}".format(self.hflip)
        print "\tvflip: {}".format(self.vflip)
        print "\tnum_classes: {}".format(self.num_classes)

    def _reshape(self, im):
        assert im is not None
        out = create_fixed_image_shape(im, frame_size=self.reshape,
                                       random_fill=True, mode='fit')
        assert out.shape == self.reshape, (
            "[ERROR] Transformer: Reshape unsuccessful, resized img shape %s and"
            " `reshape` %s" % (out.shape, self.reshape))
        return out

    def _zoom(self, im):
        low_scale, high_scale = self.zoom
        scale_comp = self.rng.choice(
            np.arange(low_scale, high_scale,
                      (high_scale - low_scale) / 100), 1)[0]
        res = cv2.resize(im, None, fx=scale_comp, fy=scale_comp)
        out = self._zoom_reshape(res, im.shape)
        return out

    def _zoom_reshape(self, im, size):
        if im.shape == size or not size:
            return im
        h, w = im.shape[0], im.shape[1]
        ho, wo = size[0], size[1]

        if h <= ho or w <= wo:
            frame_size = (ho, wo, im.shape[2]) if im.ndim == 3 else (ho, wo)
            out = np.random.randint(
                0, high=255, size=frame_size).astype(np.uint8)
            left_h, left_w = int((ho - h) / 2), int((wo - w) / 2)
            out[left_h:left_h + h, left_w:left_w + w, ...] = im
            return out

        range_h, range_w = h - ho, w - wo

        left_h = int(range_h / 2)
        left_w = int(range_w / 2)
        out = im[left_h:left_h + ho, left_w:left_w + wo, ...]
        return out

    def _vflip(self, im):
        if self.rng.rand(1) > 0.5:
            return im[::-1, ...]
        else:
            return im

    def _hflip(self, im):
        if self.rng.rand(1) > 0.5:
            return im[:, ::-1, ...]
        else:
            return im

    def _random_crop(self, im, bool_train):
        h, w = im.shape[0], im.shape[1]
        ho, wo = self.random_crop[0], self.random_crop[1]

        if h <= ho or w <= wo:
            frame_size = (ho, wo, im.shape[2]) if im.ndim == 3 else (ho, wo)
            out = create_fixed_image_shape(
                im, frame_size, random_fill=True, fill_val=0, mode='fit')
            return out

        range_h, range_w = h - ho, w - wo

        left_h = (self.rng.randint(0, range_h, 1)
                  if bool_train else int(range_h / 2))
        left_w = (self.rng.randint(0, range_w, 1)
                  if bool_train else int(range_w / 2))
        out = im[left_h:left_h + ho, left_w:left_w + wo, ...]
        return out

    def _mean_normalize(self, im):
        mean = im.mean()
        out = im.astype(np.float32) - mean
        # out /= 255.
        return out

    def transform(self, x, y, mode='test'):
        ''' Performs the selected augmentations on input image.

        Parameters:
            x:      array like, image
            y:      int, label
            mode:   augmentations performed only if 'train' else
                    'zoom', 'hflip' and 'vflip' are switched off
                    and 'random_crop' returns the center crop.
        Order of execution as follow:
            1. reshape
            2. zoom
            3. random crop
            4. random hflip
            5. random vflip
        Returns the one hot vector for y.

        Override this function to implement custom transformations.
        '''
        out_x = None
        out_y = None
        try:
            bool_train = True if mode.lower() == 'train' else False

            reshaped = self._reshape(x) if self.reshape else x
            cropped = (self._random_crop(reshaped, bool_train)
                       if self.random_crop else reshaped)
            zoomed = (self._zoom(cropped)
                      if self.zoom and bool_train else cropped)
            hflipped = (self._hflip(zoomed)
                        if self.hflip and bool_train else zoomed)
            vflipped = (self._vflip(hflipped)
                        if self.vflip and bool_train else hflipped)
            out_x = self._mean_normalize(vflipped)
            # print out_x.mean(), out_x.dtype
            # print out_x
            out_y = (one_hot(y, self.num_classes)
                     if self.num_classes is not None else y)
            if self.out_shape:
                assert out_x.shape[:2] == self.out_shape[:2], (
                    "[ERROR] Transformer: transformed image shape ({})"
                    " and expected model input shape ({}) mismatch.".format(
                        out_x.shape, self.out_shape))
        except Exception as e:
            print "[ERROR] Transformer:"
            print "\t", e
            raise e
        return (out_x, out_y)
