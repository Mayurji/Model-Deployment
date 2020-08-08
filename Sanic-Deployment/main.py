#Install Sanic
"""
pip install sanic
"""
import os

from sanic import Sanic
from sanic import response
from http import HTTPStatus
from sanic.exceptions import NotFound
from sanic_cors import CORS
from handler.routes import server
from model.model import NN_Model_K
import torch

"""
Create a Sanic App, app gets exposed when you run this file.
"""


app = Sanic(__name__)
CORS(app)

app.blueprint(server)

@app.listener('before_server_start')
async def init(app, loop):
	def load_model():
		model = NN_Model_K(n_channel=32) #Model takes the output channel for first Conv Layer from user.
		model.load_state_dict(torch.load("model/" + 'birds_vs_airplanes.pt', map_location='cpu'))

		return model

	app.train_model = load_model()


@app.listener('after_server_stop')
async def close_connection(app, loop):
	pass


@app.exception(NotFound)
async def ignore_404s(request, exception):
    return response.json({'status': HTTPStatus.NOT_FOUND, 'message': 'Route not found'})


"""
This conditional statement is executed when this script is executed. Since Sanic is a server, when you execute this file, it will run port 8000.
"""
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, workers=4, access_log=False)
