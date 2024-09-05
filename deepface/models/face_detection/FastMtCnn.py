import torch
# import time
from typing import Any, List, Dict, Tuple, Union
import numpy as np
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.modules.preprocessing import resize_image


class FastMtCnnClient(Detector):
    """
    Fast MTCNN face detector client using PyTorch and CUDA if available.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()

    def detect_faces(
        self, img: List[Union[np.ndarray, torch.Tensor]]
    ) -> List[Dict[str, List[FacialAreaRegion]]]:
        """
        Detect and align faces with MTCNN.

        Args:
            img (List[Union[np.ndarray, torch.Tensor]]): List of pre-loaded images as numpy arrays or torch tensors.

        Returns:
            List[Dict[str, List[FacialAreaRegion]]]: A list of dictionaries containing FacialAreaRegion objects.
        """
        # self.save_images(img, "original")
        # print("Image Device in detect_faces:", img[0].device)
        max_height = max(
            (
                image.shape[-2]
                if isinstance(image, torch.Tensor)
                else (image.shape[0] if isinstance(image, np.ndarray) else 0)
            )
            for image in img
        )
        max_width = max(
            (
                image.shape[-1]
                if isinstance(image, torch.Tensor)
                else (image.shape[1] if isinstance(image, np.ndarray) else 0)
            )
            for image in img
        )
        common_dim = (max_height, max_width)

        img_tensors = []
        original_dims = []

        for original_image in img:
            if isinstance(original_image, np.ndarray):
                original_dims.append(original_image.shape[:2])
                image_tensor = (
                    torch.from_numpy(original_image)
                    .permute(2, 0, 1)
                    .float()
                    .to(self.device)
                )
            elif isinstance(original_image, torch.Tensor):
                original_dims.append(original_image.shape[1:3])
                image_tensor = original_image.to(self.device)
            else:
                raise TypeError(f"Unsupported image type: {type(original_image)}")

            image_tensor = resize_image(image_tensor, common_dim)
            img_tensors.append(image_tensor)

        batch_tensor = torch.stack(img_tensors)

        # Convert from [B, C, H, W] to [B, H, W, C] for MTCNN
        batch_tensor = batch_tensor.permute(0, 2, 3, 1)

        # start_time = time.time() * 1000
        with torch.no_grad():
            detections_batch = self.model.detect(batch_tensor, landmarks=True)
        # print("Detection Time:", time.time() * 1000 - start_time)

        resp_batch = []
        if detections_batch is not None and len(detections_batch) > 0:
            for idx, (
                regions_batch,
                confidence_batch,
                eyes_batch,
                original_dim,
            ) in enumerate(zip(*detections_batch, original_dims)):
                resp = []
                if (
                    regions_batch is None
                    or confidence_batch is None
                    or eyes_batch is None
                ):
                    resp_batch.append({"faces": []})
                    continue

                scale_x = original_dim[1] / common_dim[1]
                scale_y = original_dim[0] / common_dim[0]

                for regions, confidence, eyes in zip(
                    regions_batch, confidence_batch, eyes_batch
                ):
                    if regions is None or confidence is None or eyes is None:
                        continue

                    # Convert to list if it's a numpy array
                    regions = (
                        regions.tolist() if isinstance(regions, np.ndarray) else regions
                    )
                    regions = [
                        float(r) for r in regions
                    ]  # Ensure all elements are float

                    x1, y1, x2, y2 = regions
                    x = x1 * scale_x
                    y = y1 * scale_y
                    w = (x2 - x1) * scale_x
                    h = (y2 - y1) * scale_y

                    eyes = eyes.tolist() if isinstance(eyes, np.ndarray) else eyes
                    eyes = [[float(coord) for coord in eye] for eye in eyes]
                    right_eye = tuple(
                        map(int, [eyes[0][0] * scale_x, eyes[0][1] * scale_y])
                    )
                    left_eye = tuple(
                        map(int, [eyes[1][0] * scale_x, eyes[1][1] * scale_y])
                    )

                    facial_area = FacialAreaRegion(
                        x=int(x),
                        y=int(y),
                        w=int(w),
                        h=int(h),
                        left_eye=left_eye,
                        right_eye=right_eye,
                        confidence=float(confidence),
                    )
                    resp.append(facial_area)

                resp_batch.append({"faces": resp})

        # images_with_detections = self.draw_detections(img, resp_batch)
        # Save images with detections
        # self.save_images(images_with_detections, "face_detected")
        return resp_batch

    @staticmethod
    def xyxy_to_xywh(regions: List[float]) -> List[float]:
        """
        Convert bounding box from (x1, y1, x2, y2) format to (x, y, w, h) format.

        Args:
            regions (List[float]): Bounding box in (x1, y1, x2, y2) format.

        Returns:
            List[float]: Bounding box in (x, y, w, h) format.
        """
        x1, y1, x2, y2 = regions
        return [x1, y1, x2 - x1, y2 - y1]

    def build_model(self) -> Any:
        """
        Build a fast MTCNN face detector model.

        Returns:
            Any: MTCNN model instance.
        """
        try:
            from facenet_pytorch import MTCNN as fast_mtcnn
        except ModuleNotFoundError as e:
            raise ImportError(
                "FastMtcnn is an optional detector, ensure the library is installed. "
                "Please install using 'pip install facenet-pytorch'"
            ) from e

        return fast_mtcnn(device=self.device)

    def draw_detections(
        self,
        img: List[Union[np.ndarray, torch.Tensor]],
        detections: List[Dict[str, List[FacialAreaRegion]]],
    ) -> List[np.ndarray]:
        """
        Draw bounding boxes and eye markers on the images.

        Args:
            img (List[Union[np.ndarray, torch.Tensor]]): List of original images.
            detections (List[Dict[str, List[FacialAreaRegion]]]): Detection results.

        Returns:
            List[np.ndarray]: List of images with bounding boxes and eye markers drawn.
        """
        import cv2

        output_images = []

        for original_image, detection in zip(img, detections):
            if isinstance(original_image, torch.Tensor):
                original_image = original_image.permute(1, 2, 0).cpu().numpy()
            image_with_detections = original_image.copy()

            for face in detection["faces"]:
                x, y, w, h = face.x, face.y, face.w, face.h
                left_eye = face.left_eye
                right_eye = face.right_eye

                # Draw bounding box
                cv2.rectangle(
                    image_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2
                )

                # Draw eyes
                cv2.circle(image_with_detections, left_eye, 2, (0, 0, 255), 2)
                cv2.circle(image_with_detections, right_eye, 2, (0, 0, 255), 2)

            output_images.append(image_with_detections)

        return output_images

    def save_images(self, img: List[Union[np.ndarray, torch.Tensor]], prefix: str):
        """
        Save images to disk.

        Args:
            img (List[Union[np.ndarray, torch.Tensor]]): List of images to save.
            prefix (str): Prefix for the filenames.
        """
        import cv2

        for idx, image in enumerate(img):
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()
            filename = f"{prefix}_image_{idx}.jpg"
            cv2.imwrite(filename, image[..., ::-1])  # Convert RGB to BGR for OpenCV
