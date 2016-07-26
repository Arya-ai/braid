import numpy as np
import cv2


def create_fixed_image_shape(img, frame_size=(200, 200, 3), random_fill=True,
                             fill_val=0, mode='fit'):
    # if mode == 'fit':
    X1, Y1 = frame_size[1], frame_size[0]
    image_frame = np.ones(frame_size, dtype=np.uint8) * fill_val
    if random_fill:
        image_frame = np.random.randint(
            0, high=255, size=frame_size).astype(np.uint8)

    if ((img.ndim == 2 or img.shape[2] == 1) and
            (len(frame_size) == 3 and frame_size[2] == 3)):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    X2, Y2 = img.shape[1], img.shape[0]

    if float(X1) / Y1 >= float(X2) / Y2:
        scale = float(Y1) / Y2
    else:
        scale = float(X1) / X2

    img = cv2.resize(img, None, fx=scale, fy=scale)
    sx, sy = img.shape[1], img.shape[0]

    yc = int(round((frame_size[0] - sy) / 2.))
    xc = int(round((frame_size[1] - sx) / 2.))
    image_frame[yc:yc + sy, xc:xc + sx] = img
    assert image_frame.shape == frame_size

    return image_frame


def one_hot(y, num_classes):
    Y = np.zeros((num_classes,), dtype=np.uint8)
    Y[y] = 1
    return Y
