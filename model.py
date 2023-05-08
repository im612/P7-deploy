# Da https://www.youtube.com/watch?v=h5wLuVDr0oc
# https://github.com/AssemblyAI-Examples/ml-fastapi-docker-heroku/

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import streamlit as st
# from pydantic import BaseModel
from itertools import chain
import os

BASE_DIR = Path(__file__).resolve(strict=True).parent


def load_colnames():
    global colnames
    colnames = pd.read_csv(f"{BASE_DIR}/backend/colnames.csv").columns.to_list()
    return colnames

def get_df():
    colnames = load_colnames()

    # Section liste numéros clients
    access_id = os.environ['S3_KEY']
    access_key = os.environ['S3_SECRET']
    aws_bucket = 'p7-bucket'
    global df
    df = pd.read_csv(f"s3://{aws_bucket}/test_split_orig.csv",  storage_options={'key': access_id, 'secret': access_key})
    # https: // s3fs.readthedocs.io / en / latest / api.html # s3fs.core.S3FileSystem
    # https: // devcenter.heroku.com / articles / config - vars
    return df


def load_indnames():
    df = get_df()
    indnames = pd.DataFrame(df, columns=['SK_ID_CURR']).astype(int).values
    # del df
    merged = list(chain.from_iterable(indnames.tolist()))
    return merged

#
#
# def load_indnames():
#     test_df = load_testdf()
#     indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values
#     del test_df
#     merged = list(chain.from_iterable(indnames.tolist()))
#     return merged
#
#
# def get_indnames():
#     colnames, test_df, indnames = load_data()
#     del colnames
#     del test_df
#     # >>> list2d = [[1,2,3], [4,5,6], [7], [8,9]]
#     merged = list(chain.from_iterable(indnames.tolist()))
#     return merged







#



def load_testdf():
    test_df = pd.read_csv(f"{BASE_DIR}/backend/test_split_orig2.csv")
    colnames = load_colnames()
    test_df = pd.DataFrame(test_df, columns=colnames)
    del colnames
    test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
    return test_df




def load_data():
    colnames = pd.read_csv(f"{BASE_DIR}/backend/colnames.csv").columns.to_list()
    test_df = pd.read_csv(f"{BASE_DIR}/backend/test_split_orig2.csv")
    test_df = pd.DataFrame(test_df, columns=colnames)
    test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
    indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values

    return colnames, test_df, indnames

#
def load_x():
    test_df = load_testdf()
    X = test_df.drop(columns='TARGET')
    del test_df
    return X


# def load_data():
#     colnames = pd.read_csv(f"{BASE_DIR}/colnames.csv").columns.to_list()
#
#     test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig2.csv")
#     # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig.csv")
#     test_df = pd.DataFrame(test_df, columns=colnames)
#     test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
#
#     indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values
#
#     return colnames, test_df, indnames


# def load_data():
#     colnames = pd.read_csv(f"{BASE_DIR}/colnames.csv").columns.to_list()
#
#     test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig2.csv")
#     # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig.csv")
#     test_df = pd.DataFrame(test_df, columns=colnames)
#     test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
#
#     indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values
#
#     return colnames, test_df, indnames
#



# print(get_indnames())

#
# colnames = pd.read_csv(f"{BASE_DIR}/colnames.csv").columns.to_list()
#
# # test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig.csv")
# test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig2.csv")
# test_df = pd.DataFrame(test_df, columns=colnames)
# test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)

# X = test_df.drop(columns='TARGET')

#Modèle
with open(f"{BASE_DIR}/estimator_HistGBC_Wed_Mar_22_23_35_47_2023.pkl", "rb") as f:
    model = pickle.load(f)
f.close()


def get_line( id ):
    id = int(id)
    X = load_x()
    X_line = pd.DataFrame(X.loc[X['SK_ID_CURR'] == id])
    X_line = X_line.drop(columns='SK_ID_CURR')
    return X_line


def get_the_rest():
    best_model = model
    X_work = load_x()
    threshold = 0.9
    return best_model, X_work, threshold

def get_threshold():
    best_model, X_work, threshold = get_the_rest()
    return threshold


def get_indice( id ):
    best_model, X_work, threshold = get_the_rest()
    id = int(id)
    # ind_line = X_work.loc[X_work['SK_ID_CURR'] == id].index[0]
    pd.DataFrame(X.loc[X['SK_ID_CURR'] == id])
    ind_line = X_work.loc[X_work['SK_ID_CURR'] == id].index
    return ind_line


def get_probability_df(id):
    best_model, X, threshold = get_the_rest()
    X_line = get_line(id)
    output_prob = best_model.predict_proba(X_line)
    output_prob = pd.DataFrame(output_prob)
    output_prob.rename(columns={0: 'P0', 1: 'P1'}, inplace=True)
    prob_P1 = float(output_prob['P1'].to_list()[0])

    return prob_P1


# def get_probability_df(best_model, id, X, threshold):
def get_prediction(id):
    best_model, X, threshold = get_the_rest()
    # X_line = get_line(id, X)
    X_line = get_line(id)
    output_prob = best_model.predict_proba(X_line)
    output_prob = pd.DataFrame(output_prob)
    output_prob.rename(columns={0: 'P0', 1: 'P1'}, inplace=True)
    prob_P1 = output_prob['P1'].to_list()[0]

    if prob_P1 < threshold:
        prediction = 0
    else:
        prediction = 1

    return prediction


# Interpetabilité
# Explainer
def get_explainer():
    # Explainer
    # with open(f"{BASE_DIR}/explainer.pkl", "rb") as f:
    with open(f"{BASE_DIR}/model_frontend/explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    f.close()
    return explainer


# with open(f"{BASE_DIR}/model_frontend/explainer.pkl", "rb") as f:
#     explainer = pickle.load(f)
# f.close()


# def run_shap(id):
#     best_model, X, threshold = get_the_rest()
#     explainer = get_explainer()
#     ind_line = get_ind(id, X)
#
#     shap_values = explainer.shap_values(X)
#
#     fig = shap.summary_plot(shap_values, X, show=False)
#     plt.savefig('shap_global.png')
#
#     fig1 = shap.plots.waterfall(shap_values[ind_line])
#     plt.savefig('shap_local.png')
#     plt.close()
