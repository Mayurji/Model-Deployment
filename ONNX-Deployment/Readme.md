
## **What is ONNX?**

**Open Neural Network Exchange (ONNX) is an open standard format for representing machine learning models. 
ONNX is supported by a community of partners who have implemented it in many frameworks and tools.**

## **Why ONNX?**

Enabling interoperability between different frameworks and streamlining the path from research to production 
helps increase the speed of innovation in the AI community. ONNX helps to solve the challenge of hardware dependency 
related to AI models and enables deploying same AI models to several HW accelerated targets.

Use ONNX Converter Image to convert other major model frameworks to ONNX. Supported frameworks are currently

  * CNTK, CoreML, Keras, scikit-learn, Tensorflow, PyTorch.

### **Packages**

* onnx==1.7.0
* torchvision==0.7.0
* numpy==1.18.4
* torch==1.6.0
* onnxruntime==1.4.0
* caffe2==0.8.1
* Pillow==7.2.0

### **Installation**
```python
pip install -r requirements.txt
```

### **Note**

**The purpose of introducing ONNX, Consider I have built a multiple models in pytorch and tensorflow then for deploying 
i need to set up both pytorch and tensorflow in my environment to make inference. So to avoid this, once a model is 
converted to ONNX, you can deploy the model using sanic or through docker containers in any environment which supports
ONNX.**

**Interestingly, there is a wide support for ONNX in many devices including Edge(IOT) and Mobile.**

