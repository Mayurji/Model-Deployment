from fastapi import FastAPI, File, UploadFile
from model.preprocessing import Prediction
import torch

app =  FastAPI()
predict = Prediction()

@app.get("/")
async def home():
	return {"message": "FastAPI Home"}


@app.post("/inference")
async def inference(file: UploadFile = File(...)):
	
	y_pred = predict.run_inference(await file.read())

	return {"Class Name": y_pred}

if __name__ == '__main__':

	uvicorn.run(app, host='127.0.0.1', port=8000)
