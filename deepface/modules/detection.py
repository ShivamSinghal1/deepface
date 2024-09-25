import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional, Union

from kornia.geometry.transform import rotate

from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.modules import modeling
from deepface.commons import image_utils


def save_images(img_list: List[Union[np.ndarray, torch.Tensor]], prefix: str):
    """
    Save images to disk for debugging.

    Args:
        img_list (List[Union[np.ndarray, torch.Tensor]]): List of images to save.
        prefix (str): Prefix for the filenames.
    """
    import cv2

    for idx, img in enumerate(img_list):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        filename = f"{prefix}_image_{idx}.jpg"
        cv2.imwrite(filename, img[..., ::-1])  # Convert RGB to BGR for OpenCV


def extract_faces(
    img_tensors: List[torch.Tensor],
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Extract faces from given images.

    Args:
        img_tensors (List[torch.Tensor]): List of image tensors.
        detector_backend (str): Face detector backend.
        align (bool): Align faces.
        expand_percentage (int): Expand detected facial area.
        anti_spoofing (bool): Perform anti-spoofing check.
        max_faces (Optional[int]): Maximum number of faces to process per image.

    Returns:
        List[Dict[str, List[Dict[str, Any]]]]: List of dictionaries containing detected faces and their properties.
    """
    base_regions = [
        FacialAreaRegion(x=0, y=0, w=img.shape[2], h=img.shape[1], confidence=0)
        for img in img_tensors
    ]

    # start_time = time.time() * 1000
    face_objs_batch = detect_faces(
        detector_backend=detector_backend,
        img_batch=img_tensors,
        align=align,
        expand_percentage=expand_percentage,
        max_faces=max_faces,
    )
    # print("Time taken in detect_faces:", time.time() * 1000 - start_time)

    resp_objs_batch = []
    all_faces = []
    all_facial_areas = []
    all_img_indices = []

    for img_index, (face_objs, img_tensor, base_region) in enumerate(
        zip(face_objs_batch, img_tensors, base_regions)
    ):
        face_objs = face_objs["faces"]
        resp_objs = []

        for face_obj in face_objs:
            current_img = face_obj.img
            current_region = face_obj.facial_area

            if current_img.shape[1] == 0 or current_img.shape[2] == 0:
                continue

            x = max(0, int(current_region.x))
            y = max(0, int(current_region.y))
            w = min(base_region.w - x - 1, int(current_region.w))
            h = min(base_region.h - y - 1, int(current_region.h))

            resp_obj = {
                "face": current_img,
                "facial_area": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "left_eye": current_region.left_eye,
                    "right_eye": current_region.right_eye,
                },
                "confidence": round(float(current_region.confidence or 0), 2),
            }

            if anti_spoofing:
                all_faces.append(img_tensor)
                all_facial_areas.append((x, y, w, h))
                all_img_indices.append((img_index, len(resp_objs)))

            resp_objs.append(resp_obj)

        resp_objs_batch.append({"faces": resp_objs})

    if anti_spoofing and all_faces:
        antispoof_model = modeling.build_model(task="spoofing", model_name="Fasnet")
        # start_time = time.time() * 1000
        antispoof_results = antispoof_model.analyze(
            imgs=all_faces, facial_areas=all_facial_areas
        )
        # print("Time taken in antispoof_model.analyze:", time.time() * 1000 - start_time)

        for (img_index, face_index), (is_real, antispoof_score) in zip(
            all_img_indices, antispoof_results
        ):
            resp_objs_batch[img_index]["faces"][face_index]["is_real"] = is_real
            resp_objs_batch[img_index]["faces"][face_index][
                "antispoof_score"
            ] = antispoof_score

    return resp_objs_batch


def detect_faces(
    detector_backend: str,
    img_batch: List[torch.Tensor],
    align: bool = True,
    expand_percentage: int = 0,
    max_faces: Optional[int] = None,
) -> List[DetectedFace]:
    face_detector: Detector = modeling.build_model(
        task="face_detector", model_name=detector_backend
    )

    new_img_batch = []
    original_sizes = []

    # Save original images for debugging
    # save_images(img_batch, "original_images")

    for idx, img in enumerate(img_batch):
        _, height, width = img.shape
        height_border = int(0.5 * height)
        width_border = int(0.5 * width)
        if align:
            img = F.pad(
                img,
                (width_border, width_border, height_border, height_border),
                mode="constant",
                value=0,
            )
        new_img_batch.append((img, (width_border, height_border)))
        original_sizes.append((height, width))

        # Save padded images for debugging
        # save_images([img], f"padded_image_{idx}")

    facial_areas_batch = face_detector.detect_faces([img for img, _ in new_img_batch])
    # print("Time taken in detect_faces:", time.time() * 1000 - start_time)

    all_facial_areas = []
    all_imgs = []
    all_borders = []
    all_original_sizes = []
    for idx, (
        facial_areas,
        (img, (width_border, height_border)),
        (height, width),
    ) in enumerate(zip(facial_areas_batch, new_img_batch, original_sizes)):
        facial_areas = facial_areas["faces"]

        # Handle max_faces per image
        if max_faces is not None and max_faces < len(facial_areas):
            facial_areas = sorted(
                facial_areas,
                key=lambda fa: fa.w * fa.h,
                reverse=True,
            )[:max_faces]

        all_facial_areas.extend(facial_areas)
        all_imgs.extend([img] * len(facial_areas))
        all_borders.extend([(width_border, height_border)] * len(facial_areas))
        all_original_sizes.extend([(height, width)] * len(facial_areas))

        # Save images with detections for debugging
        # Here you can draw bounding boxes on images and save them
        detected_faces = [
            {"x": fa.x, "y": fa.y, "w": fa.w, "h": fa.h} for fa in facial_areas
        ]
        # Implement a function to draw these boxes if needed

    results = expand_and_align_face_batch(
        all_facial_areas,
        all_imgs,
        align,
        expand_percentage,
        all_borders,
        all_original_sizes,
    )
    # print("Time taken in expand_and_align_face_batch:", time.time() * 1000 - start_time)

    resp_batch = []
    index = 0
    for facial_areas in facial_areas_batch:
        faces = []
        for _ in facial_areas["faces"]:
            faces.append(results[index])
            index += 1
        resp_batch.append({"faces": faces})

    return resp_batch


def expand_and_align_face_batch(
    facial_areas: List[FacialAreaRegion],
    imgs: List[torch.Tensor],
    align: bool,
    expand_percentage: int,
    borders: List[Tuple[int, int]],
    original_sizes: List[Tuple[int, int]],
) -> List[DetectedFace]:
    device = imgs[0].device

    detected_faces = []

    for i, (fa, img, border, original_size) in enumerate(
        zip(facial_areas, imgs, borders, original_sizes)
    ):
        x = torch.tensor(float(fa.x), device=device)
        y = torch.tensor(float(fa.y), device=device)
        w = torch.tensor(float(fa.w), device=device)
        h = torch.tensor(float(fa.h), device=device)
        conf_i = torch.tensor(fa.confidence, device=device)
        left_eye = torch.tensor(
            fa.left_eye if fa.left_eye is not None else (0.0, 0.0), device=device
        )
        right_eye = torch.tensor(
            fa.right_eye if fa.right_eye is not None else (0.0, 0.0), device=device
        )

        width_border, height_border = border

        if expand_percentage > 0:
            expanded_w = w + (w * expand_percentage / 100)
            expanded_h = h + (h * expand_percentage / 100)

            x = torch.clamp(x - ((expanded_w - w) / 2), min=0)
            y = torch.clamp(y - ((expanded_h - h) / 2), min=0)
            w = expanded_w
            h = expanded_h

        if align and (left_eye.sum() != 0 and right_eye.sum() != 0):
            # Adjust eye coordinates by adding borders
            left_eye += torch.tensor([width_border, height_border], device=device)
            right_eye += torch.tensor([width_border, height_border], device=device)

            # Calculate angle
            dY = left_eye[1] - right_eye[1]
            dX = left_eye[0] - right_eye[0]
            angle = torch.atan2(dY, dX) * 180 / math.pi

            # Center for rotation
            center = torch.tensor([img.shape[2] / 2, img.shape[1] / 2], device=device)

            # Rotate image
            img_rotated = rotate(
                img.unsqueeze(0),
                angle.view(1),
                center=center.view(1, 2),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            # Save rotated images for debugging
            # save_images([img_rotated], f"rotated_image_{i}")

            # Project facial area
            img_size = torch.tensor(
                [img.shape[2], img.shape[1]], device=device, dtype=torch.float32
            )

            facial_area = torch.stack([x, y, x + w, y + h], dim=0)
            rotated_coords = project_facial_area(facial_area, angle, img_size, center)

            x1, y1, x2, y2 = rotated_coords

            x1, y1, x2, y2 = (
                int(round(x1.item())),
                int(round(y1.item())),
                int(round(x2.item())),
                int(round(y2.item())),
            )
            face = img_rotated[:, y1:y2, x1:x2]

            # Save detected face images for debugging
            # save_images([face], f"detected_face_{i}")

            # Adjust facial areas back after removing borders
            x -= width_border
            y -= height_border
            left_eye -= torch.tensor([width_border, height_border], device=device)
            right_eye -= torch.tensor([width_border, height_border], device=device)
        else:
            # If alignment is not performed or eye coordinates are not available
            face = img[:, int(y) : int(y + h), int(x) : int(x + w)]

            # Save detected face images for debugging
            # save_images([face], f"detected_face_{i}")

        detected_faces.append(
            DetectedFace(
                img=face,
                facial_area=FacialAreaRegion(
                    x=int(round(x.item())),
                    y=int(round(y.item())),
                    w=int(round(w.item())),
                    h=int(round(h.item())),
                    confidence=conf_i.item(),
                    left_eye=(
                        tuple(left_eye.cpu().numpy().astype(int).tolist())
                        if left_eye.sum() != 0
                        else None
                    ),
                    right_eye=(
                        tuple(right_eye.cpu().numpy().astype(int).tolist())
                        if right_eye.sum() != 0
                        else None
                    ),
                ),
                confidence=conf_i.item(),
            )
        )

    return detected_faces


def project_facial_area(
    facial_area: torch.Tensor,
    angle: torch.Tensor,
    size: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    device = facial_area.device

    # Convert angle to radians
    angle_rad = angle * math.pi / 180

    # Rotation matrix
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    rotation_matrix = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=device
    )

    # Center of the facial area
    facial_center = torch.stack(
        [
            (facial_area[0] + facial_area[2]) / 2,
            (facial_area[1] + facial_area[3]) / 2,
        ]
    )

    # Translate center to origin
    translated_center = facial_center - center

    # Rotate the center
    rotated_center = torch.matmul(rotation_matrix, translated_center)

    # Translate back
    rotated_center += center

    # Width and height of the facial area
    width_half = (facial_area[2] - facial_area[0]) / 2
    height_half = (facial_area[3] - facial_area[1]) / 2

    # Calculate new coordinates
    x1 = rotated_center[0] - width_half
    y1 = rotated_center[1] - height_half
    x2 = rotated_center[0] + width_half
    y2 = rotated_center[1] + height_half

    # Clamp coordinates to image boundaries
    img_width, img_height = size[0], size[1]

    x1 = torch.clamp(x1, min=0, max=img_width)
    y1 = torch.clamp(y1, min=0, max=img_height)
    x2 = torch.clamp(x2, min=0, max=img_width)
    y2 = torch.clamp(y2, min=0, max=img_height)

    return torch.stack([x1, y1, x2, y2], dim=0)
