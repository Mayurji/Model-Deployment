from http import HTTPStatus
from sanic import Blueprint, response
from handler.server import Prediction

server = Blueprint('server')

"""
'/home' - it is a route of our app. Through this route, we can implement the functions we need, based on our requirement, here it is returning string "Welcome to Home".
A route can be different types like GET, POST, PUT, DELETE etc.  By default, its GET request. '/home' is dummy get request to check if everything is working fine with flask.
"""
@server.get('/hello', strict_slashes=True)
async def home(request):
    return response.json({'status': HTTPStatus.OK, 'message': 'Hello .. '})

"""
For Inference, We will need an route which accepts a POST request.
Because POST request takes in parameters from user unlike GET, which just sits around & show same results. 
Like an Login Page of a website.
"""
@server.post('/inference', strict_slashes=True)
async def predict(request):	
	if 'file' not in request.files:
		return response.json({'status': HTTPStatus.BAD_REQUEST, 'message': 'image is required'})

	
	file = request.files.get('file')
	file_stream = file.body
	train_model = request.app.train_model
	p = Prediction(train_model)
	#img_bytes = img.read()
	class_name = p.run_inference(image=file_stream)

	return response.json({ 'class_name': class_name}) 