import os
import torch
from typing import List, Any, Union
import gdown
import numpy as np
from deepface.commons import folder_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()


class FaceNet512dONNXClient(FacialRecognition):
    """
    FaceNet-512d ONNX model class
    """

    def __init__(self):
        self.model = load_facenet512d_onnx_model()
        self.model_name = "FaceNet-512d-onnx"
        self.input_shape = (160, 160)
        self.output_shape = 512

    def forward(self, img: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Generate embedding for the input image.

        Args:
            img (Union[np.ndarray, torch.Tensor]): Input image or batch of images.

        Returns:
            List[float]: Embedding vector.
        """
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        # # Convert PyTorch tensor to numpy array if necessary
        # if isinstance(img, torch.Tensor):
        #     img = img.cpu().numpy()

        # Ensure the input is in the correct format (NHWC for ONNX models)
        # if len(img.shape) == 3:
        #     img = np.expand_dims(img, axis=0)
        # elif len(img.shape) == 4:
        #     if img.shape[1] == 3:  # NCHW format
        #         img = np.transpose(img, (0, 2, 3, 1))  # Convert to NHWC
        # else:
        #     raise ValueError(f"Unexpected input shape: {img.shape}")

        result = self.model.run([output_name], {input_name: img})
        return result[0]


def load_facenet512d_onnx_model(
    url: str = "https://github.com/ShivamSinghal1/deepface/releases/download/v1/facenet512_fp32.onnx",
) -> Any:
    """
    Download Facenet512d ONNX model weights and load.

    Args:
        url (str): URL to download the model weights.

    Returns:
        Any: Loaded ONNX model.
    """
    try:
        import torch
        import onnxruntime as ort
    except ModuleNotFoundError as e:
        raise ImportError(
            "FaceNet512ONNX is an optional model, ensure the library is installed. "
            "Please install using 'pip install onnxruntime' or "
            "'pip install onnxruntime-gpu' to use gpu"
        ) from e

    if torch.cuda.is_available():
        logger.info("Using ONNX GPU for inference")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    home = folder_utils.get_deepface_home()
    onnx_model_path = os.path.join(
        home, ".deepface/weights/facenet512_onnx_weights.onnx"
    )

    if not os.path.isfile(onnx_model_path):
        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
        logger.info(f"{os.path.basename(onnx_model_path)} will be downloaded...")
        gdown.download(url, onnx_model_path, quiet=False)

    model = ort.InferenceSession(onnx_model_path, providers=providers)
    return model
