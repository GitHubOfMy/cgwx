import os
import torch
import numpy as np
from PIL import Image
import folder_paths

#定义一个类CgwxNode，告诉程序你想使用我这这个节点，节点的名字叫啥
class CgwxNode3:
    #类装饰器，代表下面的INPUT_TYPES是类方法。
    @classmethod
    #固定函数名 INPUr_TYPES，ComfyUI 强制识别，告诉 ComfyUI:这个节点有哪些输入项、输入类型、UI参数。
    #参数 cls 代表当前类本身，约定写法。这个节点的输入信息都有啥
    def INPUT_TYPES (cls):
        # 读取input文件夹图片列表，实现和官方加载图像一样的下拉+上传按钮
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        #输入信息的具体实现
        return {
            "required":{
                #输入端口有个一个text，这个text类型是字符串
                "text": ("STRING",{
                    #这个text可以写很多行，不是只能写一行
                    "multiline": True,
                    #鼠标悬停在这个text上面的时候，显示的提示文字
                    "tooltip":"Please enter the prompt word!@才鬼顽仙cgwx"
                }),
                #这个节点要能输入一个图片，并且可以上传"image_upload"图片
                "image": (sorted(files), {"image_upload": True})
            }
        }

    #类常量，固定名称。定义节点输出端口的数据类型，顺序一一对应。
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")

    #上一个行"IIAGE"，"MASK"，"STRING"对应显示成"图片"，"遮罩"，"提示词"，不写就显示成"IMAGE"，"MASK"，"STRING"
    RETURN_NAMES = ("Image", "Mask","Prompt")
    
    #打开节点列表CGWX/01-Image&prompt，就能找到这个节点。
    CATEGORY = "CGWX/01-Image&prompt"
    
    #节点描述
    DESCRIPTION = "Load Images and prompt simultaneously @才鬼顽仙cgwx"
    
    #ComfyUI 节点**强制配置项**
    FUNCTION = "execute"
    #def后面的名字要和上一行一样，节点主执行函数self：类实例，固定参数；image, text上面required的参数
    def execute(self, image, text):
        #打开并转换成Comfyui能识别的图像
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)

        #判断是否存在Alpha透明通道
        has_alpha = img.mode == "RGBA"
        # 转RGB用于输出图像（不破坏原图对象）PIL 图片
        img_rgb = img.convert("RGB")

        # PIL 图片 → numpy 数组，shape `[H,W,3]`，数值范围 `0~255 uint8`。
        img_np = np.array(img_rgb)
        #umpy 数组转为 PyTorch tensor；转 float 并且除以 255，把像素值域从 `0~255` 归一化成 ComfyUI 标准 `0.0~1.0`。
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        #unsqueeze(0) 在最前面增加 batch 维度。形状从 `[H,W,3]` → `[1,H,W,3]`，这是 ComfyUI IMAGE 标准维度格式。
        img_tensor = img_tensor.unsqueeze(0)

        # 判断遮罩
        if has_alpha:
            # 取出遮罩RGBA 图片转 numpy，取第 4 个通道（索引 3）即 Alpha 通道。得到二维数组 `[H,W]`，数值 0 (完全透明) ~ 255 (不透明)。
            alpha = np.array(img)[:, :, 3]
            #Alpha 通道转为 tensor，归一化到 `0~1`。alpha 值含义：`0=透明`，`1=不透明`。
            alpha_tensor = torch.from_numpy(alpha).float() / 255.0
            # ⭐重要！ComfyUI绘图遮罩是反向的：白色绘图区域=mask值0
            mask_tensor = 1.0 - alpha_tensor
            #增加 batch 维度，`[H,W]` → `[1,H,W]`，ComfyUI MASK 标准格式。# shape [1,H,W]
            mask_tensor = mask_tensor.unsqueeze(0) 
        else:
            # 没有透明通道，返回全黑空白遮罩
            mask_tensor = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]))
        #输出图片，遮罩，提示词
        return (img_tensor, mask_tensor, text)
    
    #固定用法，工作流一打开就判断这个图片是否存在
    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"The image file does not exist: {image} @才鬼顽仙cgwx"
        return True
        