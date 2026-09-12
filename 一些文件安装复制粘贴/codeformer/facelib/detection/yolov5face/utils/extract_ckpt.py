import sys

import torch
import os
import folder_paths

#sys.path.insert(0, "./facelib/detection/yolov5face")
#model = torch.load("facelib/detection/yolov5face/yolov5n-face.pt", map_location="cpu")["model"]
#torch.save(model.state_dict(), "weights/facelib/yolov5n-face.pth")
sys.path.insert(0, os.path.join(folder_paths.models_dir, "codeFormer-pip/weights/facelib/detection/yolov5face"))
model = torch.load(os.path.join(folder_paths.models_dir, "codeFormer-pip/weights/facelib/detection/yolov5face/yolov5n-face.pt"), map_location="cpu")["model"]
torch.save(model.state_dict(), os.path.join(folder_paths.models_dir, "codeFormer-pip/weights/facelib/yolov5n-face.pth"))
