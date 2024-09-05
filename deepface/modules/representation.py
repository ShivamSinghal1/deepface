import torch

# import time
from typing import Any, Dict, List, Optional

from deepface.modules import modeling, detection, preprocessing


def represent(
    img_tensors: List[torch.Tensor],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    anti_spoofing: bool = False,
    embedding_normalize: bool = False,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (List[torch.Tensor]): List of image tensors.
        model_name (str): Model for face recognition.
        enforce_detection (bool): If no face is detected, raise an exception.
        detector_backend (str): Face detector backend.
        align (bool): Perform alignment based on eye positions.
        expand_percentage (int): Expand detected facial area with a percentage.
        normalization (str): Normalize the input image before feeding it to the model.
        anti_spoofing (bool): Flag to enable anti-spoofing.
        max_faces (Optional[int]): Set a limit on the number of faces to be processed.
        embedding_normalize (bool): Normalize the embedding vectors.

    Returns:
        List[Dict[str, List[Dict[str, Any]]]]: List of dictionaries containing embeddings and facial areas.
    """
    # _start_time = time.time() * 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = modeling.build_model(task="facial_recognition", model_name=model_name)

    target_size = model.input_shape

    # start_time = time.time() * 1000
    img_objs_batch = detection.extract_faces(
        img_tensors=img_tensors,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )
    # print("Extract Faces Time:", time.time() * 1000 - start_time)

    resp_objs_batch = []
    batch_images = []
    batch_facial_areas = []
    for img_objs in img_objs_batch:
        for img_obj in img_objs["faces"]:
            if anti_spoofing and not img_obj.get("is_real", True):
                continue
            img = img_obj["face"]  # This is now a tensor
            img = preprocessing.resize_image(img, target_size)
            img = img / 255.0
            img = img.permute(1, 2, 0)
            batch_images.append(img)
            batch_facial_areas.append(img_obj["facial_area"])

    embedding = []
    if batch_images:
        batch_tensor = torch.stack(batch_images).cpu().numpy()
        # print("Batch Tensor Shape:", batch_tensor.shape)
        # start_time = time.time() * 1000
        with torch.no_grad():
            embedding = model.forward(batch_tensor)
        # print("Embedding Model Forward Time:", time.time() * 1000 - start_time)

        if embedding_normalize:
            embedding = torch.nn.functional.normalize(
                torch.from_numpy(embedding), p=2, dim=1
            )
            embedding = embedding.numpy()

    idx = 0

    for img_objs in img_objs_batch:
        resp_objs = []
        for img_obj in img_objs["faces"]:
            img_obj.pop("face")
            if anti_spoofing and not img_obj.get("is_real", True):
                resp_objs.append(
                    {
                        "embedding": [],
                        "facial_area": img_obj,
                    }
                )
                continue
            resp_objs.append(
                {
                    "embedding": embedding[idx].tolist(),
                    "facial_area": img_obj,
                }
            )
            idx += 1

        resp_objs_batch.append({"faces": resp_objs})

    # print("Total Time Taken in Represent:", time.time() * 1000 - _start_time)
    return resp_objs_batch
