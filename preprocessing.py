"""
preprocessing.py
================
Pipeline tiền xử lý ảnh đáy mắt (fundus) cho khâu suy luận (Inference Pipeline).
Hỗ trợ loại bỏ viền đen, resize giữ tỷ lệ (letterbox), lọc nhiễu Ben Graham và chuẩn hóa Tensor PyTorch.
"""

from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"

import io
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
# Disable multi-threading memory overhead in PyTorch safely
try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

torch.set_grad_enabled(False)
from torchvision import transforms

TARGET_SIZE = (224, 224)
CROP_TOLERANCE = 12
BEN_SIGMA = 10
BLACK_BORDER_RATIO_THRESH = 0.05
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_fundus_circle(img: np.ndarray, tolerance: int = CROP_TOLERANCE) -> np.ndarray:
    """Crop bounding box của vùng sáng trên ảnh BGR (loại viền đen)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, mask = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y: y + h, x: x + w]


def auto_detect_border(img: np.ndarray, thresh: float = BLACK_BORDER_RATIO_THRESH) -> bool:
    """Tự động phát hiện xem ảnh có viền đen xung quanh hay không dựa trên tỷ lệ pixel tối."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float((gray < CROP_TOLERANCE).mean()) > thresh


def letterbox_resize(
    img: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Resize ảnh về kích thước target_size giữ nguyên aspect ratio (đệm viền đen)."""
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    pad_y = (th - nh) // 2
    pad_x = (tw - nw) // 2
    canvas[pad_y: pad_y + nh, pad_x: pad_x + nw] = resized
    return canvas


def ben_graham_transform(img: np.ndarray, sigma_x: int = BEN_SIGMA) -> np.ndarray:
    """Xử lý tăng cường tương phản Ben Graham: output = 4*img - 4*Blur + 128."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_x)
    enhanced = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def full_preprocess_pipeline(
    img: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    use_ben_graham: bool = True,
    force_crop: bool | None = None,
) -> np.ndarray:
    """
    Pipeline xử lý ảnh OpenCV đầy đủ:
    1. Tự động kiểm tra & Crop viền đen
    2. Resize letterbox về target_size
    3. Ben Graham contrast enhancement (nếu use_ben_graham=True)
    Returns: numpy array BGR
    """
    do_crop = auto_detect_border(img) if force_crop is None else force_crop
    if do_crop:
        img = crop_fundus_circle(img)
    img = letterbox_resize(img, target_size)
    if use_ben_graham:
        img = ben_graham_transform(img)
    return img


def load_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Chuyển đổi các định dạng đầu vào (Path, Bytes, PIL Image, BGR Numpy Array) -> OpenCV BGR array.
    """
    if isinstance(image_input, (str, bytes, bytearray)):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Không thể đọc file ảnh từ đường dẫn: {image_input}")
            return img
        else:
            buf = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Không thể giải mã dữ liệu bytes thành ảnh.")
            return img
    elif isinstance(image_input, Image.Image):
        # PIL (RGB) -> OpenCV (BGR)
        img_rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # Grayscale
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        return image_input.copy()
    else:
        raise TypeError(f"Kiểu dữ liệu đầu vào không được hỗ trợ: {type(image_input)}")


def prepare_image_tensor(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    use_ben_graham: bool = True,
) -> torch.Tensor:
    """
    Nhận đầu vào linh hoạt -> Tiền xử lý OpenCV -> Chuyển thành PyTorch Tensor (1, C, H, W).
    """
    img_bgr = load_image(image_input)
    processed_bgr = full_preprocess_pipeline(
        img_bgr, target_size=target_size, use_ben_graham=use_ben_graham
    )
    # OpenCV BGR -> PIL RGB -> PyTorch Tensor
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(processed_rgb)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tensor = transform(pil_img)  # shape: (3, H, W)
    return tensor


"""
model.py
========
Định nghĩa kiến trúc mô hình EfficientNet-B4 + CBAM (Convolutional Block Attention Module)
phục vụ bài toán phân loại Mức độ Bệnh Võng mạc Tiểu đường (Diabetic Retinopathy - 5 lớp ICDR).
"""


import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    """
    Channel Attention Sub-module của CBAM.
    Tính toán trọng số chú ý cho từng kênh đặc trưng dựa trên AvgPool và MaxPool.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Sub-module của CBAM.
    Tập trung vào các vùng không gian quan trọng (xuất huyết, vi phình mạch, xuất tiết).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Kết hợp Channel Attention và Spatial Attention nối tiếp.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class EfficientNetB4_CBAM(nn.Module):
    """
    Kiến trúc mô hình chính:
    - Backbone: EfficientNet-B4 (Feature Extractor: 1792 channels)
    - Attention: CBAM Module đặt ngay sau feature extractor
    - Global Pooling: AdaptiveAvgPool2d(1)
    - Classifier: Dropout(0.3) + Linear(1792 -> 5 classes)
    """
    def __init__(
        self,
        num_classes: int = 5,
        drop_rate: float = 0.3,
        cbam_ratio: int = 16,
        pretrained: bool = False,
    ):
        super().__init__()
        if pretrained:
            weights = models.EfficientNet_B4_Weights.DEFAULT
            backbone = models.efficientnet_b4(weights=weights)
        else:
            backbone = models.efficientnet_b4(weights=None)

        self.features = backbone.features
        in_planes = 1792  # Output channel size of EfficientNet-B4 features
        self.cbam = CBAM(in_planes, ratio=cbam_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_planes, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(
    num_classes: int = 5,
    drop_rate: float = 0.3,
    cbam_ratio: int = 16,
    pretrained: bool = False,
) -> EfficientNetB4_CBAM:
    """Hàm helper khởi tạo mô hình và in số lượng tham số."""
    model = EfficientNetB4_CBAM(
        num_classes=num_classes,
        drop_rate=drop_rate,
        cbam_ratio=cbam_ratio,
        pretrained=pretrained,
    )
    return model



import os
import json
import base64
from typing import Optional, Tuple
import torch.nn.functional as F

class AuthenticEfficientNetGradCAM:
    """
    Authentic Grad-CAM for EfficientNet architectures using PyTorch forward and full backward hooks.
    Target layer: EfficientNet Features final block or CBAM.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "cbam"):
                target_layer = model.cbam
            elif hasattr(model, "features"):
                target_layer = model.features[-1]
            else:
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, (nn.Conv2d, nn.Sequential)):
                        target_layer = module
                        break

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.handles = []
        if self.target_layer is not None:
            h_fwd = self.target_layer.register_forward_hook(self._forward_hook)
            h_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)
            self.handles.extend([h_fwd, h_bwd])

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        self.model.zero_grad()
        x = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            cam = np.ones((x.shape[2], x.shape[3]), dtype=np.float32)
            return cam, target_class, float(probs[0, target_class].item())

        grads = self.gradients.cpu().data.numpy()[0]  # [C, H, W]
        acts = self.activations.cpu().data.numpy()[0]  # [C, H, W]

        weights = np.mean(grads, axis=(1, 2))  # [C]
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        cam = np.maximum(cam, 0)  # ReLU
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        h, w = x.shape[2], x.shape[3]
        cam = cv2.resize(cam, (w, h))
        return cam, target_class, float(probs[0, target_class].item())

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


class DRPredictor:
    def __init__(self, weights_path: str | None = None, device: str | None = None):
        self.class_names = {
            "0": "No DR",
            "1": "Mild",
            "2": "Moderate",
            "3": "Severe",
            "4": "Proliferative DR"
        }
        self.img_size = (224, 224)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.drop_rate = 0.3
        self.cbam_ratio = 16
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Disable multi-threading memory overhead in PyTorch safely
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
            
        if weights_path is None:
            weights_path = "efficientnet_b4_cbam_fold1.pth"
            
        if not os.path.exists(weights_path):
            print(f"[INFO] Weights file '{weights_path}' not found. Downloading dynamically from Hugging Face...")
            from huggingface_hub import hf_hub_download
            hf_token = os.getenv("HF_TOKEN")
            weights_path = hf_hub_download(
                repo_id="chrisnguyenx/EfficientNet-P3",
                filename="efficientnet_b4_cbam_fold1.pth",
                token=hf_token
            )
            
        self.model = EfficientNetB4_CBAM(
            num_classes=5,
            drop_rate=self.drop_rate,
            cbam_ratio=self.cbam_ratio,
            pretrained=False,
        )
        
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, mmap=True, weights_only=False)
        except Exception:
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, mmap=True)
            except Exception:
                checkpoint = torch.load(weights_path, map_location=self.device)
            
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        del checkpoint
        import gc
        gc.collect()
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Grad-CAM engine
        self.cam_engine = AuthenticEfficientNetGradCAM(self.model)
        
        # Free memory immediately to avoid Render OOM
        del state_dict
        import gc
        gc.collect()

    def predict(self, image_input, use_ben_graham: bool = True):
        tensor_img = prepare_image_tensor(
            image_input=image_input,
            target_size=self.img_size,
            mean=self.mean,
            std=self.std,
            use_ben_graham=use_ben_graham,
        )
        batch_tensor = tensor_img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        # Run Grad-CAM with gradients enabled locally
        with torch.enable_grad():
            cam, _, _ = self.cam_engine.generate_cam(batch_tensor, target_class=pred_class)
            
        # Generate BGR image for overlay
        img_bgr = load_image(image_input)
        processed_bgr = full_preprocess_pipeline(
            img_bgr, target_size=(224, 224), use_ben_graham=use_ben_graham
        )
        
        # Apply JET color map to CAM heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Overlay heatmap with the preprocessed image
        overlay = cv2.addWeighted(processed_bgr, 0.6, heatmap, 0.4, 0)
        
        # Encode overlay to base64
        _, encoded_img = cv2.imencode(".jpg", overlay)
        base64_gradcam = base64.b64encode(encoded_img).decode("utf-8")
        gradcam_base64_str = f"data:image/jpeg;base64,{base64_gradcam}"
        
        probabilities_dict = {
            self.class_names.get(str(i), f"Class {i}"): float(probs[i])
            for i in range(len(probs))
        }
        
        return {
            "class_id": pred_class,
            "class_name": self.class_names.get(str(pred_class), f"Class {pred_class}"),
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "gradcam_image_base64": gradcam_base64_str,
        }
