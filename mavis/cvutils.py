import io
from pathlib import Path
from typing import Union

import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter1d as gauss, median_filter
from skimage.filters import sobel
from skimage.morphology import skeletonize
from sklearn.metrics import pairwise_distances_argmin

from pilutils import pil
from db import ConfigDAO


def load_label(path, inverted=False):
    label = Image.open(path)
    label = np.array(label.convert("L")).astype(np.uint8)
    if inverted:
        label = 255 - label
    label[label <= 10] = 0
    label[label > 10] = 1
    return label


def to_watershed_markers(path, inverted=False, dist_threshold=0):
    # Convert the label into [1, 0] Images
    sure_bg = load_label(path, inverted)
    # st.image(pil(sure_bg*255))

    markers = watershed_from_background(sure_bg, dist_threshold)
    # st.image(pil(markers))
    return markers


def watershed_from_background(sure_bg, dist_threshold, flood=False):
    # Finding sure foreground probability by distance to each contour border
    # st.image(pil(sure_bg))

    dist_transform = cv.distanceTransform(sure_bg, cv.DIST_L2, 5)
    # st.image(pil(dist_transform*255))
    # Threshold foreground based on distance to border for each contour
    ret, sure_fg = cv.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    # st.image(pil(sure_fg*255))
    # Find all unknown regions
    if flood:
        unknown = cv.subtract(np.ones_like(sure_bg), sure_fg)
    else:
        unknown = cv.subtract(sure_bg, sure_fg)
    # st.image(pil(unknown*255))
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    # st.image(pil(markers))
    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0
    return markers


def to_contours(markers,
                std_dev=2,
                out_image=None,
                contour_color=(255, 0, 0),
                contour_thickness=1,
                calculate_properties=True):
    """
    Converts opencv watershed markers to contours.
    Optionally draws contours in an image.
    Optionally calculates properties of each contour

    :param markers: np.ndarray Opencv Watershed Markers
    :param out_image: Image to draw the contours, if None, no contours are drawn
    :param contour_color: Contour Color
    :param std_dev: Gaussian Smoothing of the contour
    :param calculate_properties: Flag whether to calculate opencv contour properties
    :return:
        - modified out_image or None if out_image is None
        - DataFrame with Contour Properties or empty DataFrame if calculate_properties is False
    """
    properties = []

    for i in range(np.max(markers) - 1):
        cnt_image = (markers == i + 2).astype(np.uint8)
        for cnt in cv.findContours(cnt_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]:
            if cnt is None:
                continue

            if std_dev:
                cnt = gauss(np.asarray(cnt, dtype=np.int32), std_dev, axis=0, mode="wrap")

            if out_image is not None:
                cv.drawContours(out_image, [cnt], 0, contour_color, contour_thickness)

            if calculate_properties:
                properties += [contour_properties(cnt)]

    return out_image, pd.DataFrame(properties)


def contour_properties(cnt):
    """
    Calculate opencv contour properties

    :param cnt: The opencv contour (list of coordinates)
    :return: A dictionary with the properties
    """
    res = cv.moments(cnt)
    if res["m00"] != 0:
        res["Centroid X"] = int(res['m10'] / res['m00'])
        res["Centroid Y"] = int(res['m01'] / res['m00'])

    res["Area"] = cv.contourArea(cnt)
    res["Perimeter"] = cv.arcLength(cnt, True)
    res["Convex"] = cv.isContourConvex(cnt)

    x, y, w, h = cv.boundingRect(cnt)
    res["Bounding Box (x,y,w,h)"] = (x, y, w, h)
    res["Aspect Ration (Bounding Box)"] = float(w) / h

    hull = cv.convexHull(cnt)
    res["Convex Hull Area"] = cv.contourArea(hull)

    if res["Convex Hull Area"] > 0:
        res["Solidity"] = float(res["Area"]) / res["Convex Hull Area"]

    res["Equi. Diameter"] = np.sqrt(4 * res["Area"] / np.pi)

    if res["Area"] > 5:
        (cx, cy), (MA, ma), angle = cv.fitEllipse(cnt)

        res.update({
            "Ellips. Center X": cx,
            "Ellips. Center Y": cy,
            "Ellips. Major Axis": MA,
            "Ellips. Minor Axis": ma,
            "Ellips. Angle": angle
        })

        res["Circularity"] = 4 * np.pi * res["Area"] / res["Perimeter"] ** 2

    return res


def contour_border_check(img_np, cnt):
    rect = cv2.boundingRect(cnt)
    w, h = rect[2] + 2, rect[3] + 2
    return np.array_equiv(contour_roi(img_np, rect, pad=1).shape[:2], (h, w))


def contour_roi(img_np, rect, pad=0):
    r = rect
    max_h, max_w = img_np.shape[:2]
    x, y, w, h = r[0] - pad, r[1] - pad, r[2] + (pad * 2), r[3] + (pad * 2)
    roi = img_np[
          max(y, 0): min(y + h, max_h),
          max(x, 0): min(x + w, max_w)
          ]
    return roi


def external_contours(mask):
    img = Image.fromarray(mask)
    img = img.resize(np.array(img.size) * 2, resample=0)
    np_img = np.asarray(img).astype(np.uint8)

    scaled_cnts, hierachy = cv2.findContours(np_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    external_cnts = [np.ceil(cnt / 2).astype(np.int32) for cnt in scaled_cnts]
    return external_cnts


def circularity(cnt, a):
    return 4 * np.pi * a / max(cv2.arcLength(cnt, True) ** 2, 1)


def filtered_contours(mask, min_circularity, min_area, max_area, ignore_border_cnt=False):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ext_cnts = external_contours(mask)

    res = [
        (cnt, area, ext_cnt)
        for cnt, ext_cnt in zip(cnts, ext_cnts) for area in [cv2.contourArea(ext_cnt)] if (
                (min_circularity == 0 or circularity(ext_cnt, area) > min_circularity)
                and (min_area == 0 or min_area < area)
                and (max_area == 0 or area < max_area)
                and (not ignore_border_cnt or contour_border_check(mask, cnt))
        )
    ]

    filtered_cnts, areas, filtered_ext_cnts = zip(*res) if res else ([], [], [])
    return filtered_cnts, areas, filtered_ext_cnts


def color_coded_to_labelme(img_path: Union[Path, str],
                           seg_map: Union[str, Image.Image],
                           exclude_classes: str,
                           save_with_contents=False,
                           save_full_path=False):
    """
    Converts a coler encdoed image to labelme.json


    Parameters
    ----------
    img_path: Path to the image
    seg_map: Segmentation map to the image
    exclude_classes: str, classes to exclude, by name
    save_with_contents: Weather to save the image as byte string in the labelme.json
    save_full_path: Weather to to save the full path or relative path to image in labelme.json

    Returns
    -------

    """
    lbl = np.array(seg_map if type(seg_map) == Image else Image.open(seg_map).convert("RGB"))
    label_me = LabelMeJsonWrapper(img_path, save_with_contents, save_full_path)

    for i, col in enumerate(ConfigDAO()["CLASS_COLORS"]):
        if ConfigDAO()["CLASS_NAMES"][i] in exclude_classes:
            continue
        mask = np.zeros(lbl.shape[:-1]).astype(np.uint8)
        mask[(lbl == tuple(col)).all(axis=-1)] = 1
        label_me.add(mask, i)

    return label_me.img, label_me.labels


class LabelMeJsonWrapper:
    def __init__(self,
                 path,
                 save_with_contents=False,
                 save_full_path=False):
        """
        Wrapper for labelme files

        Parameters
        ----------
        path: the file path for the image
        save_with_contents: Wether to save the image as byte string to the labelme.json
        save_full_path: Wether to include the full path. Otherwise relative path is written to labelme.json
        """

        self.save_with_contents = save_with_contents
        self.save_full_path = save_full_path
        self.path = path
        self.img = Image.open(path).convert("RGB")
        self.labels = self._prepare()

    def _prepare(self) -> dict:
        """
        Prepares an empty labelme.json dict

        Returns
        -------
        The json as dict
        """
        if self.save_with_contents:
            with io.BytesIO() as output:
                self.img.save(output, format="JPEG")
                contents = "".join(map(chr, output.getvalue()))

        return {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": self.path if self.save_full_path else Path(self.path).name,
            "imageData": contents if self.save_with_contents else None,
            "imageWidth": self.img.size[0],
            "imageHeight": self.img.size[1],
        }

    def add(self, mask, class_id) -> None:
        """
        Adds a single object via a mask

        Parameters
        ----------
        mask: The binary mask for the object
        class_id: The class id of the object

        Returns
        -------
        None
        """
        for cnt in external_contours(mask):
            self.labels["shapes"] += [{
                "label": str(ConfigDAO()["CLASS_NAMES"][class_id]),
                "points": np.squeeze(cnt).astype(float).tolist(),
                "group_id": int(class_id),
                "shape_type": "polygon",
                "flags": {}
            }]


class ConnectedComponentCleaner:
    threshold = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    def preprocess(self, img):
        return img

    def process(self, img):
        result = cv2.bitwise_not(img)
        contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        result = np.zeros(shape=[result.shape[0], result.shape[1], 1], dtype=np.uint8)

        for element in contours:
            current = np.zeros(shape=[result.shape[0], result.shape[1], 1], dtype=np.uint8)
            cv2.drawContours(current, [element], -1, 255, -1)
            current = cv2.dilate(current, self.kernel2)
            current = cv2.erode(current, self.kernel2)
            result = cv2.bitwise_or(result, current)
        # st.image(pil(result))
        return result

    def postprocess(self, img):
        img = cv2.bitwise_not(img)
        return img

    def process_one(self, img):
        ret2, result = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        return self.postprocess(self.process(self.preprocess(result)))


class ConnectedComponentSkeletonize(ConnectedComponentCleaner):
    def preprocess(self, result):
        # st.image(pil(result))
        result = result / result.max()
        result = skeletonize(result)
        result = result * 255
        result = np.uint8(result)
        result = cv2.dilate(result, self.kernel)
        # st.image(pil(result))
        return result

    def postprocess(self, result):
        result = cv2.erode(result, self.kernel)
        result = cv2.bitwise_not(result)
        return result


class CVProcessor:
    def __init__(self):
        self.name = None
        self.i = None
        self.opts = None
        self.selection = None
        self.opt = None

    def configure_opt(self):
        st.write(f"--- \n ### Step {self.i + 1}: {self.name}")
        values = list(self.opts.keys())
        default = values.index(self.selection) if hasattr(self, "selection") and self.selection in values else 0
        self.selection = st.selectbox("Type", values, default, key=f"{self.name}{self.i}type")
        self.opt = self.opts[self.selection]()

    def run(self, gray):
        gray = np.array(gray.convert("L")).astype(np.uint8)
        return pil(self.opt(gray))


class BlurProcessor(CVProcessor):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.name = "Blur"
        self.m_blur = 5
        self.g_blur = 5
        self.opts = {
            "Median": lambda: self.median_blur_config(),
            "Gaussian": lambda: self.gaussian_blur_config(),
        }

    def gaussian_blur_config(self):
        self.g_blur = st.number_input(
            "Gaussian Blur Radius",
            1, 15, (self.g_blur + 1) // 2,
            key=f"g_blur{self.i}"
        ) * 2 - 1
        return self.gaussian_blur

    def median_blur_config(self):
        self.m_blur = st.number_input(
            "Median Blur Radius",
            1, 15, (self.m_blur + 1) // 2,
            key=f"m_blur{self.i}"
        ) * 2 - 1
        return self.median_blur

    def gaussian_blur(self, gray):
        return cv2.GaussianBlur(gray, (self.g_blur, self.g_blur), sigmaX=0, sigmaY=0)

    def median_blur(self, gray):
        return cv2.medianBlur(gray, self.m_blur)


class MorphProcessor(CVProcessor):
    class Morph:
        structuring_element_cv_types = {
            "Rectangle": cv.MORPH_RECT,
            "Cross": cv.MORPH_RECT,
            "Ellipse": cv.MORPH_ELLIPSE
        }

        morph_type_cv = {
            "Erosion": cv.MORPH_ERODE,
            "Dilation": cv.MORPH_DILATE,
            "Opening": cv.MORPH_OPEN,
            "Closing": cv.MORPH_CLOSE,
            "Top-Hat": cv.MORPH_TOPHAT,
            "Black-Hat": cv.MORPH_BLACKHAT,
            "Morphological Gradient": cv.MORPH_GRADIENT
        }

        def __init__(self, i):
            self.i = i
            self.kernel_size = 3
            self.iterations = 1
            self.structuring_element = list(MorphProcessor.Morph.structuring_element_cv_types.keys())[0]
            self.morph_type = list(MorphProcessor.Morph.morph_type_cv.keys())[0]

        def configure(self):
            st.write(f"Morph Operation {self.i + 1}")
            self.kernel_size = st.slider(
                "Kernel Radius", 0, 21,
                self.kernel_size,
                key=f"kernel_size_{self.i}"
            )
            se_values = list(MorphProcessor.Morph.structuring_element_cv_types.keys())
            se_default = se_values.index(self.structuring_element)
            self.structuring_element = st.selectbox(
                "Structuring Element",
                se_values, se_default,
                key=f"structuring_element_{self.i}"
            )
            self.iterations = st.slider(
                "Iteration",
                0, 10, self.iterations,
                key=f"iterations_{self.i}"
            )
            mt_values = list(MorphProcessor.Morph.morph_type_cv.keys())
            mt_default = mt_values.index(self.morph_type)
            self.morph_type = st.selectbox(
                "Morph Type",
                mt_values, mt_default,
                key=f"morph_type_{self.i}"
            )

        def run(self, img):
            structuring_cv_type = MorphProcessor.Morph.structuring_element_cv_types[self.structuring_element]
            element = cv.getStructuringElement(
                structuring_cv_type,
                (2 * self.kernel_size + 1, 2 * self.kernel_size + 1),
                (self.kernel_size, self.kernel_size)
            )
            return cv.morphologyEx(img, MorphProcessor.Morph.morph_type_cv[self.morph_type], element)

    def __init__(self, i):
        super().__init__()
        self.i = i
        self.name = "Morph"
        self.n_operations = 0
        self.morph_operations = []
        self.opt = self.opt_fn

    def configure_opt(self):
        st.write(f"### Step {self.i + 1}: {self.name}")
        self.n_operations = st.slider("Number of Morph Operations", 0, 5, self.n_operations)

        # if st.button("Update Morph operations"):
        self.morph_operations = [MorphProcessor.Morph(i) for i in range(self.n_operations)]

        for m in self.morph_operations:
            m.configure()

    def opt_fn(self, gray):
        for m in self.morph_operations:
            gray = m.run(gray)
        return gray


class InverterProcessor(CVProcessor):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.name = "Invert"
        self.opt = lambda gray: 255 - gray

    def configure_opt(self):
        st.write(f"### Step {self.i + 1}: {self.name}")


class ThresholdProcessor(CVProcessor):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.name = "Threshold"
        self.threshold = 127
        self.use_hsv = False
        self.d_size = 1
        self.block_size = 5
        self.opts = {
            "Binary Thresholding": lambda: self.binary_config(),
            "Adaptive Gaussian Thresholding": lambda: self.adaptive_gaussian_config(),
            "OTSU Thresholding": lambda: self.otsu
        }

    def binary_config(self):
        self.use_hsv = st.checkbox("Use HSV", self.use_hsv)
        self.threshold = st.slider(
            f"Threshold limit",
            0, 255, self.threshold,
            key=f"binary_threshold{self.i}"
        )
        return self.binary

    def adaptive_gaussian_config(self):
        self.d_size = st.number_input(
            "Block Radius",
            0, 400, self.d_size,
            key=f"adaptive_threshold{self.i}"
        )
        self.block_size = self.d_size * 2 + 1
        return self.adaptive_gaussian

    def binary(self, gray):
        ret, gray = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return gray

    def adaptive_gaussian(self, gray):
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, self.block_size, 2)
        return gray

    @staticmethod
    def otsu(gray):
        ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray


class EdgeProcessor(CVProcessor):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.name = "Edge"
        self.k_size = 5
        self.upper_threshold = 100
        self.lower_threshold = 100
        self.opts = {
            "Laplacian": lambda: self.lap_config(),
            "Canny": lambda: self.canny_config(),
            "Sobel": lambda: self.sob
        }

    def lap_config(self):
        self.k_size = st.number_input("Kernel Radius", 0, 255, (self.k_size + 1) // 2, key=f"lks_{self.i}") * 2 - 1
        return self.laplacian

    def canny_config(self):
        self.lower_threshold = st.slider("Lower Threshold", 0, 255, self.lower_threshold, key=f"clt_{self.i}"),
        self.upper_threshold = st.slider("Upper Threshold", 0, 255, self.upper_threshold, key=f"cut_{self.i}")
        return self.canny

    @staticmethod
    def normalize(gray):
        mx = gray.max()
        mask = gray > 0
        gray = (gray / mx) * 255
        gray *= mask
        return gray

    def laplacian(self, gray):
        gray = -cv2.Laplacian(gray, cv2.CV_64F, ksize=self.k_size)
        gray = self.normalize(gray)
        return gray

    def canny(self, gray):
        print(gray.shape)
        gray = cv2.Canny(image=gray, threshold1=self.lower_threshold, threshold2=self.upper_threshold)
        return gray

    @staticmethod
    def sob(gray):
        gray = sobel(gray)
        return gray


def color_compress(np_img, n_bins=4, median_kernel=10):
    h, w, d = np_img.shape
    bin_edges = np.linspace(0, 255, n_bins + 1)
    bin_centres = (bin_edges[0:-1] + bin_edges[1::]) / 2.

    hist, _ = np.histogramdd(np_img.reshape(-1, 3), bins=np.vstack(3 * [bin_edges]))

    color_codebook = np.vstack([
        [[0, 0, 0]],
        np.column_stack([bin_centres[dim] for dim in np.where(hist)]).tolist(),
        [[255, 255, 255]]
    ])

    targets = pairwise_distances_argmin(color_codebook,
                                        np_img.reshape((-1, 3)),
                                        axis=0)

    np_img = color_codebook[list(targets)].reshape((h, w, -1))
    np_img = median_filter(np_img, size=(*([median_kernel] * 2), 1))

    return np_img
