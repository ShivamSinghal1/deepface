# built-in dependencies
from typing import Any, Dict, List, Union, Optional

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.commons import image_utils
from deepface.modules import modeling, detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition


def represent(
    img_path: List[Union[str, np.ndarray]],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
    embedding_normalize: bool = False,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed (default is None).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (List[float]): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).
        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.
        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.
    """
    resp_objs_batch = []

    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = model.input_shape
    if detector_backend != "skip":
        img_objs_batch = detection.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = image_utils.load_image(img_path)

        if len(img.shape) != 3:
            raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")

        # make dummy region and confidence to keep compatibility with `extract_faces`
        img_objs_batch = [
            {
                "faces": [
                    {
                        "face": img,
                        "facial_area": {
                            "x": 0,
                            "y": 0,
                            "w": img.shape[0],
                            "h": img.shape[1],
                        },
                        "confidence": 0,
                    }
                ]
            }
        ]
    # ---------------------------------

    # if max_faces is not None and max_faces < len(img_objs):
    #     # sort as largest facial areas come first
    #     img_objs = sorted(
    #         img_objs,
    #         key=lambda img_obj: img_obj["facial_area"]["w"] * img_obj["facial_area"]["h"],
    #         reverse=True,
    #     )
    #     # discard rest of the items
    #     img_objs = img_objs[0:max_faces]

    images_batch = []
    for img_objs in img_objs_batch:
        for img_obj in img_objs["faces"]:
            if anti_spoofing is True and img_obj.get("is_real", True) is False:
                continue
            img = img_obj["face"]

            # rgb to bgr
            img = img[:, :, ::-1]

            # resize to expected shape of ml model
            img = preprocessing.resize_image(
                img=img,
                # thanks to DeepId (!)
                target_size=(target_size[1], target_size[0]),
            )

            # custom normalization
            img = preprocessing.normalize_input(img=img, normalization=normalization)
            images_batch.append(img)

    images_batch = np.squeeze(np.array(images_batch), axis=1)
    embedding = model.forward(images_batch)
    idx = 0

    for img_objs in img_objs_batch:
        resp_objs = []
        for img_obj in img_objs["faces"]:
            img_obj.pop("face")
            if anti_spoofing is True and img_obj.get("is_real", True) is False:
                resp_objs.append(
                    {
                        "embedding": [],
                        "facial_area": img_obj,
                    }
                )
                continue
            img_embedding = embedding[idx]
            idx += 1
            if embedding_normalize:
                norm = np.linalg.norm(img_embedding)
                if norm != 0:
                    img_embedding = img_embedding / norm

            if isinstance(img_embedding, np.ndarray):
                img_embedding = img_embedding.tolist()

            resp_objs.append(
                {
                    "embedding": img_embedding,
                    "facial_area": img_obj,
                }
            )

        resp_objs_batch.append({"faces": resp_objs})

    return resp_objs_batch
