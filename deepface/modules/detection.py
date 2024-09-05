# import time
import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Any, Tuple

from kornia.geometry import rotate

from deepface.modules import modeling
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion


def save_images(img: List[torch.Tensor], prefix: str):
    """
    Save images to disk.

    Args:
        img (List[torch.Tensor]): List of images to save.
        prefix (str): Prefix for the filenames.
    """
    import cv2

    for idx, image in enumerate(img):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        filename = f"{prefix}_image_{idx}.jpg"
        cv2.imwrite(filename, image[..., ::-1])  # Convert RGB to BGR for OpenCV


def extract_faces(
    img_tensors: List[torch.Tensor],
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    anti_spoofing: bool = False,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Extract faces from given images.

    Args:
        img_path (List[torch.Tensor]): List of image tensors.
        detector_backend (str): Face detector backend.
        align (bool): Align faces.
        expand_percentage (int): Expand detected facial area.
        normalize_face (bool): Normalize output face pixels.
        anti_spoofing (bool): Perform anti-spoofing check.

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
                "confidence": round(float(current_region.confidence), 2),
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
) -> List[DetectedFace]:
    face_detector: Detector = modeling.build_model(
        task="face_detector", model_name=detector_backend
    )

    new_img_batch = []
    for img in img_batch:
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

    # start_time = time.time() * 1000
    facial_areas_batch = face_detector.detect_faces([img for img, _ in new_img_batch])
    # print("Time taken in detect_faces:", time.time() * 1000 - start_time)

    all_facial_areas = []
    all_imgs = []
    all_borders = []
    for facial_areas, (img, (width_border, height_border)) in zip(
        facial_areas_batch, new_img_batch
    ):
        facial_areas = facial_areas["faces"]
        all_facial_areas.extend(facial_areas)
        all_imgs.extend([img] * len(facial_areas))
        all_borders.extend([(width_border, height_border)] * len(facial_areas))

    # start_time = time.time() * 1000
    results = expand_and_align_face_batch(
        all_facial_areas, all_imgs, align, expand_percentage, all_borders
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
) -> List[DetectedFace]:
    # batch_size = len(facial_areas)
    device = imgs[0].device
    # print("Device in expand_and_align_face_batch:", device)

    # Find max dimensions
    max_height = max(img.shape[1] for img in imgs)
    max_width = max(img.shape[2] for img in imgs)

    # save_images(imgs, "original")

    # Pad images to max dimensions
    padded_imgs = []
    for img in imgs:
        _, h, w = img.shape
        pad_h = max_height - h
        pad_w = max_width - w
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
        padded_imgs.append(padded_img)

    # save_images(padded_imgs, "padded")
    padded_imgs = torch.stack(padded_imgs)

    x = torch.tensor([fa.x for fa in facial_areas], device=device)
    y = torch.tensor([fa.y for fa in facial_areas], device=device)
    w = torch.tensor([fa.w for fa in facial_areas], device=device)
    h = torch.tensor([fa.h for fa in facial_areas], device=device)
    confidence = torch.tensor([fa.confidence for fa in facial_areas], device=device)

    left_eyes = torch.tensor([fa.left_eye for fa in facial_areas], device=device)
    right_eyes = torch.tensor([fa.right_eye for fa in facial_areas], device=device)

    if expand_percentage > 0:
        expanded_w = w + (w * expand_percentage // 100)
        expanded_h = h + (h * expand_percentage // 100)

        x = torch.clamp(x - ((expanded_w - w) // 2), min=0)
        y = torch.clamp(y - ((expanded_h - h) // 2), min=0)
        w = torch.min(
            torch.tensor([img.shape[2] for img in imgs], device=device) - x, expanded_w
        )
        h = torch.min(
            torch.tensor([img.shape[1] for img in imgs], device=device) - y, expanded_h
        )

    if align:
        aligned_imgs, angles = align_img_wrt_eyes_batch(
            padded_imgs, left_eyes, right_eyes
        )
        # save_images(aligned_imgs, "aligned")
        img_sizes = torch.tensor(
            [(img.shape[2], img.shape[1]) for img in imgs], device=device
        )  # Note the order change: width, height

        rotated_coords = project_facial_area_batch(
            torch.stack([x, y, x + w, y + h], dim=1), angles, img_sizes
        )
        detected_faces = [
            aligned_imgs[i, :, int(y1) : int(y2), int(x1) : int(x2)]
            for i, (x1, y1, x2, y2) in enumerate(rotated_coords)
        ]

        width_borders, height_borders = zip(*borders)
        x = x - torch.tensor(width_borders, device=device)
        y = y - torch.tensor(height_borders, device=device)

        left_eyes = left_eyes - torch.tensor(borders, device=device)
        right_eyes = right_eyes - torch.tensor(borders, device=device)

    else:
        detected_faces = [
            padded_imgs[i, :, int(y_i) : int(y_i + h_i), int(x_i) : int(x_i + w_i)]
            for i, (x_i, y_i, w_i, h_i) in enumerate(zip(x, y, w, h))
        ]
        # save_images(detected_faces, "detected_before_align")

    return [
        DetectedFace(
            img=face,
            facial_area=FacialAreaRegion(
                x=int(x_i),
                y=int(y_i),
                w=int(w_i),
                h=int(h_i),
                confidence=conf_i,
                left_eye=tuple(le_i.int().tolist()),
                right_eye=tuple(re_i.int().tolist()),
            ),
            confidence=conf_i,
        )
        for face, x_i, y_i, w_i, h_i, conf_i, le_i, re_i in zip(
            detected_faces, x, y, w, h, confidence, left_eyes, right_eyes
        )
    ]


def align_img_wrt_eyes_batch(
    imgs: torch.Tensor,
    left_eyes: torch.Tensor,
    right_eyes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs = imgs / 255.0

    # Calculate angles
    dY = right_eyes[:, 1] - left_eyes[:, 1]
    dX = right_eyes[:, 0] - left_eyes[:, 0]
    angles = torch.atan2(dY, dX) * 180 / math.pi

    # Adjust angles to align eyes horizontally
    angles = torch.where(angles > 90, angles - 180, angles)

    # Calculate centers
    centers = (left_eyes + right_eyes) / 2

    # Perform rotation using kornia
    aligned_imgs = rotate(imgs, angles, center=centers)

    aligned_imgs = aligned_imgs * 255.0
    return aligned_imgs, angles


def project_facial_area_batch(
    facial_areas: torch.Tensor, angles: torch.Tensor, sizes: torch.Tensor
) -> torch.Tensor:
    device = facial_areas.device
    batch_size = facial_areas.shape[0]

    # Normalize angles
    directions = torch.where(
        angles >= 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device)
    )
    angles = torch.abs(angles) % 360

    # Early return for zero angles
    zero_angles = angles == 0
    if zero_angles.all():
        return facial_areas

    # Convert to radians
    angles_rad = angles * math.pi / 180

    heights, widths = sizes[:, 0], sizes[:, 1]

    # Translate the facial areas to the center of the images
    x = (facial_areas[:, 0] + facial_areas[:, 2]) / 2 - widths / 2
    y = (facial_areas[:, 1] + facial_areas[:, 3]) / 2 - heights / 2

    # Rotate the facial areas
    cos_angles = torch.cos(angles_rad)
    sin_angles = torch.sin(angles_rad)
    x_new = x * cos_angles + y * directions * sin_angles
    y_new = -x * directions * sin_angles + y * cos_angles

    # Translate the facial areas back to the original positions
    x_new = x_new + widths / 2
    y_new = y_new + heights / 2

    # Calculate projected coordinates after alignment
    widths_half = (facial_areas[:, 2] - facial_areas[:, 0]) / 2
    heights_half = (facial_areas[:, 3] - facial_areas[:, 1]) / 2
    x1 = x_new - widths_half
    y1 = y_new - heights_half
    x2 = x_new + widths_half
    y2 = y_new + heights_half

    # Validate projected coordinates are in images' boundaries
    x1 = torch.clamp(x1, min=0)
    y1 = torch.clamp(y1, min=0)
    x2 = torch.clamp(x2, max=widths)
    y2 = torch.clamp(y2, max=heights)

    # Combine results
    result = torch.stack([x1, y1, x2, y2], dim=1)
    result = torch.where(zero_angles.unsqueeze(1), facial_areas, result)

    return result.int()
