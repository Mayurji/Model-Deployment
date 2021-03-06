import torch
import json
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms
from model.model import NN_Model_K

class_names = ['airplane', 'bird']


class Prediction:
	
	def __init__(self):
		self.model = NN_Model_K(n_channel=32) #Model takes the output channel for first Conv Layer from user.
		self.model.load_state_dict(torch.load("model/" + 'birds_vs_airplanes.pt', map_location='cpu'))

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