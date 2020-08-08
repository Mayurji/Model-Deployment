import caffe2.python.onnx.backend as backend
import onnxruntime
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

session = onnxruntime.InferenceSession("model/birds_vs_airplanes.onnx")
#prepare_backend = backend.prepare(model)
class_names = ['airplane', 'bird']
test_image = "sample_image.jpg"

#Transformation on Image
my_transforms = transforms.Compose([transforms.ToTensor(),
	                                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

image = Image.open(test_image)
transformed_image  = my_transforms(image).unsqueeze(0)

ort_inputs = {session.get_inputs()[0].name: transformed_image.numpy()}
ort_outs = session.run(None, ort_inputs)
img_out_y = ort_outs[0]
print(class_names[np.argmax(img_out_y[0])])
