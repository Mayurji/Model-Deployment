import requests


image = '/home/mayur/Desktop/Model_Deployment/sample_image.jpg'

resp = requests.post("http://0.0.0.0:8000/predict",
                     files={"file": open(image,'rb')})

# extracting the response
print("{}".format(resp.text))