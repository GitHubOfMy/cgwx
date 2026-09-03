import torch
import numpy as np
from PIL import Image,ImageDraw

class CgwxDrawBbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "tooltip": "边界框左上角X坐标"}),
                "y": ("INT", {"default": 0, "min": 0, "tooltip": "边界框左上角Y坐标"}),
                "width": ("INT", {"default": 512, "min": 1, "tooltip": "边界框宽度"}),
                "height": ("INT", {"default": 512, "min": 1, "tooltip": "边界框高度"}),
                "color": ("COLOR", {"default": "#ff0000", "tooltip":"边框颜色，拾取器选颜色"}),
                "line_width": ("INT", {"default":3, "min":1, "max":20, "tooltip":"边框线粗细像素"})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "CGWX/02‑Draw"
    DESCRIPTION = "Draw Bounding Box，节点内直接设置xy宽高 @才鬼顽仙cgwx"
    FUNCTION = "draw"

    def draw(self, image, x, y, width, height, color, line_width):
        # tensor -> PIL Image 工具函数
        def tensor2pil(tensor_img):
            arr = tensor_img.cpu().numpy()
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr[0])

        def pil2tensor(pil_img):
            arr = np.array(pil_img).astype(np.float32)/255.0
            return torch.from_numpy(arr).unsqueeze(0)

        pil_img = tensor2pil(image)
        draw = ImageDraw.Draw(pil_img)
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        draw.rectangle([x1,y1,x2,y2], outline=color, width=line_width)
        out_tensor = pil2tensor(pil_img)
        return (out_tensor,)
