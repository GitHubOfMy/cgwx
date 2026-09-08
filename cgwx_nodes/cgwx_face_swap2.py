import os
import cv2
import insightface
import numpy as np
import torch
import folder_paths
import sys
import tempfile

# ========== codeformer-pip导入 =========
try:
    from codeformer.app import inference_app
except ImportError:
    inference_app = None
# =======================================

INSIGHTFACE_MODEL_DIR = os.path.join(folder_paths.models_dir, "insightface")
BUFFALO_L_FOLDER = os.path.join(INSIGHTFACE_MODEL_DIR, "models/buffalo_l")
SWAP_MODEL_PATH = os.path.join(INSIGHTFACE_MODEL_DIR, "inswapper_128.onnx")

FACE_APP = None
SWAP_MODEL = None
CURR_PROVIDERS = None
DET_SIZE = (640, 640)


def init_insightface(providers):
    global FACE_APP, SWAP_MODEL, CURR_PROVIDERS
    if FACE_APP is not None and SWAP_MODEL is not None and CURR_PROVIDERS == providers:
        return True

    FACE_APP = None
    SWAP_MODEL = None

    if not os.path.isdir(INSIGHTFACE_MODEL_DIR):
        raise FileNotFoundError(
            f"[CgwxFaceSwapWithCodeFormer]模型目录不存在！\n期望路径：{INSIGHTFACE_MODEL_DIR}\n"
            "请手动创建 models/insightface ，放入 buffalo_l文件夹 和 inswapper_128.onnx，禁止自动下载。"
        )
    if not os.path.isdir(BUFFALO_L_FOLDER):
        raise FileNotFoundError(
            f"[CgwxFaceSwapWithCodeFormer]找不到 buffalo_l 文件夹！\n期望路径：{BUFFALO_L_FOLDER}\n"
            "请把完整buffalo_l文件夹放到 models/insightface/models/ 下。"
        )
    if not os.path.exists(SWAP_MODEL_PATH):
        raise FileNotFoundError(
            f"[CgwxFaceSwapWithCodeFormer]找不到 inswapper_128.onnx！\n期望路径：{SWAP_MODEL_PATH}\n"
            "请手动放入该模型文件。"
        )
        
    #防止自动下载
    FACE_APP = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=INSIGHTFACE_MODEL_DIR,
        providers=providers
    )
    FACE_APP.prepare(ctx_id=-1, det_size=DET_SIZE)
    SWAP_MODEL = insightface.model_zoo.get_model(SWAP_MODEL_PATH, providers=providers)
    CURR_PROVIDERS = providers
    return True


class CgwxFaceSwapWithCodeFormer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "use_gpu": ("BOOLEAN", {"default": False, "label_on": "使用 GPU", "label_off": "使用 CPU"}),
                "enable_codeformer": ("BOOLEAN", {"default": True, "tooltip": "开启：执行CodeFormer人脸高清修复；关闭：仅输出原始换脸结果"}),
                "codeformer_fidelity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0~1；越大保留换脸原始细节，越小修复力度越强"
                }),
                "background_enhance": ("BOOLEAN", {"default": True, "tooltip": "背景增强"}),
                "face_upsample": ("BOOLEAN", {"default": True, "tooltip": "人脸超分放大"}),
                "upscale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1, "tooltip": "放大倍数"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)
    FUNCTION = "run"
    CATEGORY = "CGWX/03‑FaceSwap"
    DESCRIPTION = "Insightface换脸 + CodeFormer-pip人脸高清修复，可关闭修复，@才鬼顽仙 cgwx"

    def run(self, source_image, target_image, use_gpu, enable_codeformer, codeformer_fidelity, background_enhance, face_upsample, upscale):
        if enable_codeformer and inference_app is None:
            raise RuntimeError("启用CodeFormer-pip无法加载，请重新安装")

        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        init_insightface(providers)

        def tensor_to_cv2(img_tensor):
            img = img_tensor[0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        def cv2_to_tensor(cv_img):
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
            return tensor.unsqueeze(0)

        src_bgr = tensor_to_cv2(source_image)
        dst_bgr = tensor_to_cv2(target_image)

        src_faces = FACE_APP.get(src_bgr)
        dst_faces = FACE_APP.get(dst_bgr)

        if len(src_faces) == 0:
            raise RuntimeError("source_image 未检测到人脸")
        if len(dst_faces) == 0:
            raise RuntimeError("target_image 未检测到人脸")

        swap_bgr = SWAP_MODEL.get(dst_bgr, dst_faces[0], src_faces[0], paste_back=True)

        if enable_codeformer:
            # pip版codeformer只接受文件路径，必须写入临时文件
            tmp_fd, tmp_swap_path = tempfile.mkstemp(suffix=".jpg")
            os.close(tmp_fd)
            try:
                cv2.imwrite(tmp_swap_path, swap_bgr)
                restored_file_path = inference_app(
                    image=tmp_swap_path,
                    background_enhance=background_enhance,
                    face_upsample=face_upsample,
                    upscale=upscale,
                    codeformer_fidelity=codeformer_fidelity
                )
                if isinstance(restored_file_path, (list, tuple)):
                    restored_file_path = restored_file_path[0]

                restored_bgr = cv2.imread(restored_file_path)
                if restored_bgr is None:
                    raise RuntimeError(f"CodeFormer修复失败，无法读取输出文件:{restored_file_path}")
                out_tensor = cv2_to_tensor(restored_bgr)
            finally:
                # 无论成功失败，都删除临时文件
                if os.path.exists(tmp_swap_path):
                    os.remove(tmp_swap_path)
        else:
            out_tensor = cv2_to_tensor(swap_bgr)

        return (out_tensor,)


