import torch
import json
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms

class_names = ['airplane', 'bird']


class Prediction:
	
	def __init__(self, train_model):
		self.model = train_model

	"""
	Setting up the transformation to be applied to each image, the transformation should be same as what was applied while during training model.
	"""
	@staticmethod
	def transform_image(image_bytes=None):
	    my_transforms = transforms.Compose([transforms.ToTensor(),
	                                        transforms.Normalize(
	                                            [0.5, 0.5, 0.5],
	                                            [0.5, 0.5, 0.5])])
	    image = Image.open(BytesIO(image_bytes))
	    return my_transforms(image).unsqueeze(0)

	def run_inference(self, image):
		self.model.eval()
		in_tensor = self.transform_image(image_bytes=image)
		with torch.no_grad():
			output = self.model(in_tensor)
			_, predicted = torch.max(output, dim=1)
			out = predicted.item()
		    
		return class_names[out]


