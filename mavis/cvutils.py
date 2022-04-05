import io
from pathlib import Path
from typing import Union

import cv2
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d as gauss, median_filter
from skimage.morphology import skeletonize
from sklearn.metrics import pairwise_distances_argmin


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
    :param contour_thickness: Contour Thickness in pixel
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


def color_coded_to_labelme(
        img_path: Union[Path, str],
        seg_map: Union[str, Image.Image],
        exclude_classes: str,
        class_colors: list,
        class_names: list,
        save_with_contents=False,
        save_full_path=False,
):
    """
    Converts a coler encdoed image to labelme.json


    Parameters
    ----------
    class_colors
    class_names
    img_path: Path to the image
    seg_map: Segmentation map to the image
    exclude_classes: str, classes to exclude, by name
    save_with_contents: Weather to save the image as byte string in the labelme.json
    save_full_path: Weather to to save the full path or relative path to image in labelme.json

    Returns
    -------

    """
    lbl = np.array(seg_map if type(seg_map) == Image else Image.open(seg_map).convert("RGB"))
    label_me = LabelMeJsonWrapper(
        img_path,
        class_names=class_names,
        save_with_contents=save_with_contents,
        save_full_path=save_full_path
    )

    for i, col in enumerate(class_colors):
        if class_names[i] in exclude_classes:
            continue
        mask = np.zeros(lbl.shape[:-1]).astype(np.uint8)
        mask[(lbl == tuple(col)).all(axis=-1)] = 1
        label_me.add(mask, i)

    return label_me.img, label_me.labels


class LabelMeJsonWrapper:
    def __init__(
            self,
            path,
            class_names,
            save_with_contents=False,
            save_full_path=False,
    ):
        """
        Wrapper for labelme files

        Parameters
        ----------
        path: the file path for the image
        save_with_contents: Wether to save the image as byte string to the labelme.json
        save_full_path: Wether to include the full path. Otherwise relative path is written to labelme.json
        """
        self.class_names = class_names
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
                "label": str(self.class_names[class_id]),
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
