"""
PyTorch integrated with Caffe2 and gets it's full production pipeline. This is the pipeline used at Facebook. 
They train the model using PyTorch and deploy it using Caffe2.
"""
"""
ONNX - Open Neural Network Exchange is an open format that lets users move deep learning models between different frameworks.
For the deployment of PyTorch models, the most common way is to convert them into an ONNX format and then deploy the exported 
ONNX model using Caffe2.
"""
from model.model import NN_Model_K
import torch.utils.model_zoo as model_zoo
import onnx
import torch


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize model with the pretrained weights
model_class = NN_Model_K(n_channel=32) #Model takes the output channel for first Conv Layer from user.
state_dict = torch.load('model/birds_vs_airplanes.pt')
model_class.load_state_dict(state_dict)


dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model_class, dummy_input, "model/birds_vs_airplanes.onnx")
