import requests


image = '/home/mayur/Desktop/Model_Deployment/Deploying Using Sanic/sample_image.jpg'

resp = requests.post("http://0.0.0.0:8000/inference",
                     files={"file": open(image,'rb')})

# extracting the response
print("{}".format(resp.text))