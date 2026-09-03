import os
import cv2
import insightface
import numpy as np
import torch
import folder_paths

#更换过ComfyUI默认目录的使用以下代码
INSIGHTFACE_MODEL_DIR = os.path.join(folder_paths.models_dir, "insightface")

#按照reactor对models目录要求
BUFFALO_L_FOLDER = os.path.join(INSIGHTFACE_MODEL_DIR, "models/buffalo_l")
SWAP_MODEL_PATH = os.path.join(INSIGHTFACE_MODEL_DIR, "inswapper_128.onnx")

FACE_APP = None
SWAP_MODEL = None
#记录当前实例使用的providers，用来判断是否需要重建
CURR_PROVIDERS = None
DET_SIZE = (640, 640)

def init_insightface(providers):
    global FACE_APP, SWAP_MODEL, CURR_PROVIDERS
    # 如果当前providers和传入一致，直接复用
    if FACE_APP is not None and SWAP_MODEL is not None and CURR_PROVIDERS == providers:
        return True

    # 如果设备不一样，清空旧模型实例，重建
    FACE_APP = None
    SWAP_MODEL = None

    # 检查基础目录
    if not os.path.isdir(INSIGHTFACE_MODEL_DIR):
        raise FileNotFoundError(
            f"[InsightFace换脸]模型目录不存在！\n期望路径：{INSIGHTFACE_MODEL_DIR}\n"
            "请手动创建 models/insightface ，放入 buffalo_l文件夹 和 inswapper_128.onnx，禁止自动下载。"
        )
    # 检查buffalo_l文件夹
    if not os.path.isdir(BUFFALO_L_FOLDER):
        raise FileNotFoundError(
            f"[InsightFace换脸]找不到 buffalo_l 文件夹！\n期望路径：{BUFFALO_L_FOLDER}\n"
            "请把完整buffalo_l文件夹放到 models/insightface/models/ 下，不要自动下载。"
        )
    # 检查换脸onnx
    if not os.path.exists(SWAP_MODEL_PATH):
        raise FileNotFoundError(
            f"[InsightFace换脸]找不到 inswapper_128.onnx！\n期望路径：{SWAP_MODEL_PATH}\n"
            "请手动放入该模型文件，本节点不会自动下载。"
        )

    #构造FaceAnalysis时传入providers
    FACE_APP = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=INSIGHTFACE_MODEL_DIR,
        providers=providers
    )
    #onnxruntime后端 ctx_id 传 -1 即可，providers已经接管设备
    FACE_APP.prepare(ctx_id=-1, det_size=DET_SIZE)

    #换脸模型同样传入providers
    SWAP_MODEL = insightface.model_zoo.get_model(SWAP_MODEL_PATH, providers=providers)
    CURR_PROVIDERS = providers
    return True


class CgwxFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "use_gpu": ("BOOLEAN", {"default": False, "label_on":"使用 GPU","label_off":"使用 CPU"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("swap_image",)
    FUNCTION = "do_CgwxFaceSwap"
    CATEGORY = "CGWX/03‑FaceSwap"
    DESCRIPTION = "Simple FaceSwap 简易换脸 @才鬼顽仙 cgwx"

    def do_CgwxFaceSwap(self, source_image, target_image, use_gpu):
        # 根据开关生成provider列表；优先CUDA，失败自动回退CPU
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

        src_img_bgr = tensor_to_cv2(source_image)
        dst_img_bgr = tensor_to_cv2(target_image)

        src_faces = FACE_APP.get(src_img_bgr)
        dst_faces = FACE_APP.get(dst_img_bgr)

        if len(src_faces) == 0:
            raise RuntimeError("脸图中未检测到人脸")
        if len(dst_faces) == 0:
            raise RuntimeError("身体图中未检测到人脸")

        result_bgr = SWAP_MODEL.get(
            dst_img_bgr,
            dst_faces[0],
            src_faces[0],
            paste_back=True
        )
        out_tensor = cv2_to_tensor(result_bgr)
        return (out_tensor,)

