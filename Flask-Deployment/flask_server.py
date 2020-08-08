#Install Flask
"""
pip install flask
"""

from flask import Flask, request, jsonify
from model import NN_Model_K
import torch
import json
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms

"""
Create a Flask App, app gets exposed when you run this file.
"""


app = Flask(__name__)
model = NN_Model_K(n_channel=32) #Model takes the output channel for first Conv Layer from user.
model.load_state_dict(torch.load("model/" + 'birds_vs_airplanes.pt', map_location='cpu'))
model.eval()
class_names = ['airplane', 'bird']


"""
Setting up the transformation to be applied to each image, the transformation should be same as what was applied while during training model.
"""
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.5, 0.5, 0.5],
                                            [0.5, 0.5, 0.5])])
    image = Image.open(BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)



def run_inference(image):
	in_tensor = transform_image(image_bytes=image)
	with torch.no_grad():
		output = model(in_tensor)
		_, predicted = torch.max(output, dim=1)
		out = predicted.item()
	    
	return class_names[out]

"""
'/home' - it is a route of our app. Through this route, we can implement the functions we need, based on our requirement, here it is returning string "Welcome to Home".
A route can be different types like GET, POST, PUT, DELETE etc.  By default, its GET request. '/home' is dummy get request to check if everything is working fine with flask.
"""
@app.route("/home")
def home():
	return "Welcome to Home"


"""
For Inference, We will need an route which accepts a POST request.
Because POST request takes in parameters from user unlike GET, which just sits around & show same results. 
Like an Login Page of a website.
"""
@app.route('/predict/', methods=["POST"])
def predict():

	if request.method == 'POST':
		file = request.files['file']
		img_bytes = file.read()
		class_name = run_inference(img_bytes)
		return jsonify({ 'class_name': class_name}) 



"""
This conditional statement is executed when this script is executed. Since Flask is a server, when you execute this file, it will run port 8000.
"""
if __name__ =='__main__':
	app.run(host='0.0.0.0',port=8000)


