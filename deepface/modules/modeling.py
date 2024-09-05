# built-in dependencies
from typing import Any


def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained models as singletonish way
    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace, GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, yunet,
                fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
    Returns:
            built model class
    """

    # singleton design pattern
    global cached_models

    model_paths = {
        "facial_recognition": {
            "VGG-Face": "deepface.models.facial_recognition.VGGFace.VggFaceClient",
            "OpenFace": "deepface.models.facial_recognition.OpenFace.OpenFaceClient",
            "Facenet": "deepface.models.facial_recognition.Facenet.FaceNet128dClient",
            "Facenet512": "deepface.models.facial_recognition.Facenet.FaceNet512dClient",
            "Facenet512ONNX": "deepface.models.facial_recognition.FacenetONNX.FaceNet512dONNXClient",
            "DeepFace": "deepface.models.facial_recognition.FbDeepFace.DeepFaceClient",
            "DeepID": "deepface.models.facial_recognition.DeepID.DeepIdClient",
            "Dlib": "deepface.models.facial_recognition.Dlib.DlibClient",
            "ArcFace": "deepface.models.facial_recognition.ArcFace.ArcFaceClient",
            "SFace": "deepface.models.facial_recognition.SFace.SFaceClient",
            "GhostFaceNet": "deepface.models.facial_recognition.GhostFaceNet.GhostFaceNetClient",
        },
        "spoofing": {
            "Fasnet": "deepface.models.spoofing.FasNet.Fasnet",
        },
        "facial_attribute": {
            "Emotion": "deepface.models.demography.Emotion.EmotionClient",
            "Age": "deepface.models.demography.Age.ApparentAgeClient",
            "Gender": "deepface.models.demography.Gender.GenderClient",
            "Race": "deepface.models.demography.Race.RaceClient",
        },
        "face_detector": {
            "opencv": "deepface.models.face_detection.OpenCv.OpenCvClient",
            "mtcnn": "deepface.models.face_detection.MtCnn.MtCnnClient",
            "ssd": "deepface.models.face_detection.Ssd.SsdClient",
            "dlib": "deepface.models.face_detection.Dlib.DlibDetector.DlibClient",
            "retinaface": "deepface.models.face_detection.RetinaFace.RetinaFaceClient",
            "mediapipe": "deepface.models.face_detection.MediaPipe.MediaPipeClient",
            "yolov8": "deepface.models.face_detection.Yolo.YoloClient",
            "yunet": "deepface.models.face_detection.YuNet.YuNetClient",
            "fastmtcnn": "deepface.models.face_detection.FastMtCnn.FastMtCnnClient",
            "centerface": "deepface.models.face_detection.CenterFace.CenterFaceClient",
        },
    }

    if model_paths.get(task) is None:
        raise ValueError(f"unimplemented task - {task}")

    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in model_paths.keys()}

    if cached_models[task].get(model_name) is None:
        model_path = model_paths[task].get(model_name)
        if model_path:
            module_path, class_name = model_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            cached_models[task][model_name] = model_class()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]
