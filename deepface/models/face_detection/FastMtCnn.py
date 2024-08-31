from copy import deepcopy
from typing import Any, Union, List, Dict
import cv2
import numpy as np
from deepface.models.Detector import Detector, FacialAreaRegion

# Link -> https://github.com/timesler/facenet-pytorch
# Examples https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch


class FastMtCnnClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def detect_faces(
        self, img: List[np.ndarray]
    ) -> List[Dict[str, List[FacialAreaRegion]]]:
        """
        Detect and align face with mtcnn

        Args:
            img (List[np.ndarray]): List of pre-loaded images as numpy arrays

        Returns:
            results (List[Dict[str, List[FacialAreaRegion]]]): A list of dictionaries containing FacialAreaRegion objects
        """
        max_width = max(image.shape[0] for image in img)
        max_height = max(image.shape[1] for image in img)
        common_dim = (max_width, max_height)

        img_rgb_batch = []
        original_dims = []

        for original_image in img:
            image = deepcopy(original_image)
            original_dims.append(image.shape[:2])
            if image.shape[0:2] != common_dim:
                factor_0 = common_dim[0] / image.shape[0]
                factor_1 = common_dim[1] / image.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (
                    int(image.shape[1] * factor),
                    int(image.shape[0] * factor),
                )
                image = cv2.resize(image, dsize)

                diff_0 = common_dim[0] - image.shape[0]
                diff_1 = common_dim[1] - image.shape[1]

                # Put the base image in the middle of the padded image
                image = np.pad(
                    image,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            if image.shape[0:2] != common_dim:
                image = cv2.resize(image, common_dim)
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_rgb_batch.append(rgb_img)

        detections_batch = self.model.detect(img_rgb_batch, landmarks=True)
        resp_batch = []
        if detections_batch is not None and len(detections_batch) > 0:
            for regions_batch, confidence_batch, eyes_batch, original_dim, image in zip(
                *detections_batch, original_dims, img
            ):
                resp = []
                if regions_batch is None or confidence_batch is None or eyes_batch is None:
                    resp_batch.append({"faces": []})
                    continue
                for regions, confidence, eyes in zip(
                    regions_batch, confidence_batch, eyes_batch
                ):
                    if regions is None or confidence is None or eyes is None:
                        continue
                    scale_x = original_dim[1] / common_dim[1]
                    scale_y = original_dim[0] / common_dim[0]
                    regions = [
                        regions[0] * scale_x,
                        regions[1] * scale_y,
                        regions[2] * scale_x,
                        regions[3] * scale_y,
                    ]

                    x, y, w, h = xyxy_to_xywh(regions)
                    right_eye = tuple(int(i * scale_x) for i in eyes[0])
                    left_eye = tuple(int(i * scale_y) for i in eyes[1])

                    facial_area = FacialAreaRegion(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        left_eye=left_eye,
                        right_eye=right_eye,
                        confidence=confidence,
                    )
                    resp.append(facial_area)
                    # Draw bounding box on the image
                    # draw_bounding_box(image, (x, y, w, h))

                resp_batch.append({"faces": resp})
                # cv2.imwrite(f"image_with_faces_{len(resp_batch)}.jpg", img_bgr)

        return resp_batch

    def build_model(self) -> Any:
        """
        Build a fast mtcnn face detector model
        Returns:
            model (Any)
        """
        # this is not a must dependency. do not import it in the global level.
        try:
            from facenet_pytorch import MTCNN as fast_mtcnn
            import torch
        except ModuleNotFoundError as e:
            raise ImportError(
                "FastMtcnn is an optional detector, ensure the library is installed. "
                "Please install using 'pip install facenet-pytorch'"
            ) from e

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return fast_mtcnn(device=device)


def xyxy_to_xywh(regions: Union[list, tuple]) -> tuple:
    """
    Convert (x1, y1, x2, y2) format to (x, y, w, h) format.
    Args:
        regions (list or tuple): facial area coordinates as x, y, x+w, y+h
    Returns:
        regions (tuple): facial area coordinates as x, y, w, h
    """
    x, y, x_plus_w, y_plus_h = regions
    w = x_plus_w - x
    h = y_plus_h - y
    return (x, y, w, h)


def draw_bounding_box(image: np.ndarray, box: tuple) -> None:
    """
    Draw a bounding box on the image.
    Args:
        image (np.ndarray): The image on which to draw the bounding box.
        box (tuple): The bounding box coordinates (x, y, w, h).
    """
    x, y, w, h = box
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
