from .cgwx_nodes.cgwx_load_img_prompt import CgwxNode3
from .cgwx_nodes.cgwx_draw_bbox import CgwxDrawBbox
from .cgwx_nodes.cgwx_face_swap import CgwxFaceSwap
from .cgwx_nodes.cgwx_face_swap2 import CgwxFaceSwapWithCodeFormer

NODE_CLASS_MAPPINGS = {
    "CgwxNode": CgwxNode3,
    "CgwxDrawBbox": CgwxDrawBbox,
    "CgwxFaceSwap": CgwxFaceSwap,
    "CgwxFaceSwapWithCodeFormer": CgwxFaceSwapWithCodeFormer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CgwxNode": "Load image and prompt @才鬼顽仙cgwx",
    "CgwxDrawBbox": "Draw BBox 绘制边界框 @才鬼顽仙cgwx",
    "CgwxFaceSwap": "Simple FaceSwap 简易换脸 @才鬼顽仙cgwx",
    "CgwxFaceSwapWithCodeFormer": "FaceSwap 高清换脸 @才鬼顽仙cgwx"
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
