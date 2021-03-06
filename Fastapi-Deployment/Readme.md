## **Objective**

Learn how to Deploy model using Fastapi.

### **Where used?**

Fastapi is widely being used for production deployment of ML models. Check out the Fastapi website for reviews.

### **Packages**

* numpy==1.19.1
* torch==1.6.0
* torchvision==0.7.0
* fastapi==0.63.0
* Pillow==8.1.2

### **Installation**
```python
pip install -r requirements.txt
```
### **To run**

By default, the app runs on 127.0.0.1:8000 and *main* refers to the filename and *app* is the fastapi object, a key point of interaction.
*uvicorn* is the ASGI (Asynchronous Server Gateway Interface) built on top of uvloop and httptools. Fastapi is a API framework. I would encourage 
users to read the docs for Fastapi for further reading.

```python
uvicorn main:app --reload
```
