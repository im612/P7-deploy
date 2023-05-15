# backend/main.py

# " this is where we put the FastAPI endpoints"
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
# https://fastapi.tiangolo.com/tutorial/security/first-steps/
from model import load_indnames
from model import get_probability_df
from model import get_prediction
from model import get_threshold
from model import load_colnames
from model import get_shap
from model import get_line

app = FastAPI()


# asyncronous models
# https://asgi.readthedocs.io/en/latest/

# Da https://www.youtube.com/watch?v=h5wLuVDr0oc
# Da https://testdriven.io/blog/fastapi-streamlit/
# https://www.youtube.com/watch?v=IvHCxycjeR0 DF

app = FastAPI()

class Id(BaseModel):
    id: str


@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/indnames")
def ind_names():
    val = load_indnames()
    return {"listindnames": val}


@app.post("/probability")
def pred_prob(iddata: Id):
    proba = float(get_probability_df(int(iddata.id)))
    return {"probability": proba}


@app.post("/prediction")
def prediction(iddata: Id):
    pred = float(get_prediction(int(iddata.id)))
    return {"prediction": pred}


@app.post("/seuil")
def prediction(iddata: Id):
    val = float(get_threshold())
    return {"seuil": val}


@app.post("/colnames")
def col_names():
    val = load_colnames()
    return {"listcolnames": val}

@app.post("/shap_val")
def prediction(iddata: Id):
    pred = get_shap(id)
    return pred

@app.post("/get_line")
def prediction(iddata: Id):
    pred = get_line(id)
    return {"listline": pred}



# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8080)
#     uvicorn.run("main:app-1container-nonfunziona", host="backend", port=8080)


# This is our server. FastAPI creates two endpoints, one dummy ("/") and
# one for serving our prediction ("/{style}"). The serving endpoint takes in a name as a URL parameter.
# [We're using nine different trained models to perform style transfer, so the path parameter
# will tell us which model_frontend to choose. The image is accepted as a file over a POST request and
# sent to the inference function. Once the inference is complete, the file is stored on the local
# filesystem and the path is sent as a response.

# Next, add the following config to a new file called backend/config.py:
