import base64
from io import BytesIO

import PIL
from PIL import Image

from db import ConfigDAO

Image.MAX_IMAGE_PIXELS = None

import numpy as np

IMAGE_FILETYPE_EXTENSIONS = [".bmp", ".png", ".jpg", ".jpeg", ".tif"]
FILETYPE_EXTENSIONS = IMAGE_FILETYPE_EXTENSIONS + [".xml", ".json"]


def pil(img: np.ndarray, normalize=True, verbose=True):
    """
    Convert any 1 or 3 channel ndarray to uint8 RGB pil image

    :param img: ndarray
    :param normalize: Uses a scaler to bring image to [0,255] uint8 first
    :param verbose: Wethr to print min, max, shape and dtype of image to convert
    :return: pil image
    """
    # Show image values
    if verbose:
        print(np.max(img), np.min(img), img.shape, img.dtype)

    # Handle empty dimensions
    img = img.squeeze()

    if normalize:
        # Handle lower than 0
        img = img.astype(np.float)
        img = img - np.min([np.min(img), 0])

        # Hanlde wrong scale
        scaler = 255 if img.max() <= 3 else 1
        img = (img * scaler)

        # Handle higher than 255
        if img.max() > 255:
            img = (img.astype(np.float) / img.max() * 255)

    # Return as png
    pil_img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return pil_img


def to_integer_encoding(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    res = np.zeros_like(img)
    for i, col in enumerate(ConfigDAO()["CLASS_COLORS"]):
        res[(img == tuple(col)).all(axis=-1)] = (i, 0, 0)
    res = Image.fromarray(res, mode="RGB")
    return res


def to_color_encoding(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    res = np.zeros_like(img)
    for i in range(len(ConfigDAO()["CLASS_COLORS"])):
        res[(img == (i, 0, 0)).all(axis=-1)] = tuple(ConfigDAO()["CLASS_COLORS"][i])
    res = Image.fromarray(res, mode="RGB")
    return res


def thumbnail_inverse(img, x_min_size, y_min_size):
    if img.size[0] < x_min_size:
        wpercent = (x_min_size / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((x_min_size, hsize), Image.ANTIALIAS)

    if img.size[1] < y_min_size:
        hpercent = (y_min_size / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, y_min_size), Image.ANTIALIAS)

    return img


def segmentation_error_mask(mask_1: PIL.Image, mask_2: PIL.Image):
    """
    Calculate a seg. error mask that shows the first mask in red, the second mask in blue and the overlap in green.
    The input should be two PIL.Image which have been converted to format "L" which stands for grayscale.
    E.g. PIL.Image.open("mask.png").convert("L") or PIL.Image.fromarray(np_255_mask).convert("L")

    Parameters
    ----------
    mask_1: PIL.Image of format "L"
    mask_2: PIL.Image of format "L"

    Returns
    -------
    numpy array of format (height, width, 3), RGB values between 0 and 255
    """
    width, height = mask_1.size
    error_img = np.zeros((height, width, 3)).astype(np.uint8)
    error_img[:, :, 2] = mask_1
    error_img[:, :, 0] = mask_2
    error_img[np.logical_and(np.array(mask_1), np.array(mask_2))] = (0, 255, 0)
    return error_img


def segmentation_map_error(path_1, path_2):
    image_1 = Image.open(str(path_1)).convert("RGBA")
    image_2 = Image.open(str(path_2)).convert("RGBA")
    error_img = segmentation_error_mask(image_1.convert("L"),
                                        image_2.convert("L"))
    error_img = Image.fromarray(error_img).convert("RGBA")
    return error_img


def get_thumbnail(path):
    i = Image.open(path).convert("RGBA")
    i.thumbnail((150, 150), Image.LANCZOS)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        base64Image = base64.b64encode(buffer.getvalue()).decode()
    return base64Image


def surface_to_pil(s):
    import cairo

    cario_format = s.get_format()
    if cario_format == cairo.FORMAT_ARGB32:
        pil_mode = 'RGB'
        # Cairo has ARGB. Convert this to RGB for PIL which supports only RGB or
        # RGBA.
        argb_array = np.fromstring(bytes(s.get_data()), 'c').reshape(-1, 4)
        rgb_array = argb_array[:, 2::-1]
        pil_data = rgb_array.reshape(-1).tostring()
    else:
        raise ValueError('Unsupported cairo format: %d' % cario_format)
    pil_image = Image.frombuffer(pil_mode,
                                 (s.get_width(),
                                  s.get_height()),
                                 pil_data, "raw", pil_mode, 0, 1)
    return pil_image.convert('RGB')
