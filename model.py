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

access_id = os.getenv('S3_KEY')
access_key = os.getenv('S3_SECRET')
aws_bucket = 'p7-bucket'

def load_colnames():
    global colnames
    colnames = pd.read_csv(f"{BASE_DIR}/backend/colnames.csv").columns.to_list()
    return colnames

def get_df():
    # df = pd.read_csv(f"{BASE_DIR}/backend/test_split_orig.csv")
    df = pd.read_csv(f"s3://{aws_bucket}/test_split_orig.csv", nrows=1000, storage_options={'key': access_id, 'secret': access_key})
    colnames = load_colnames() # inclut SK_ID et TARGET
    df = pd.DataFrame(df, columns=colnames)
    return df


def load_indnames():
    df = get_df()
    print(df.index)
    indnames = pd.DataFrame(df, columns=['SK_ID_CURR']).astype(int).values
    merged = list(chain.from_iterable(indnames.tolist()))
    return merged


#Modèle
with open(f"{BASE_DIR}/estimator_HistGBC_Wed_Mar_22_23_35_47_2023.pkl", "rb") as f:
    model = pickle.load(f)
f.close()


def get_threshold():
    threshold = 0.9
    return threshold


# def load_x():
#     x = get_df()
#     x = x.drop(columns=['SK_ID_CURR', 'TARGET'])
#     return x


def get_the_rest():
    best_model = model
    # x_work = load_x()
    x_work = get_df()
    x_work = x_work.drop(columns=['SK_ID_CURR', 'TARGET'])
    threshold = get_threshold()
    return best_model, x_work, threshold


def get_line(id_i):
    idi = int(id_i)
    x = get_df()
    x = x.drop(columns=['TARGET'])
    x_line = pd.DataFrame(x.loc[x['SK_ID_CURR'] == idi])
    x_line = x_line.drop(columns='SK_ID_CURR')
    return x_line


def get_probability_df(id):
    best_model, x, threshold = get_the_rest()
    x_line = get_line(id)
    output_prob = best_model.predict_proba(x_line)
    output_prob = pd.DataFrame(output_prob)
    output_prob.rename(columns={0: 'P0', 1: 'P1'}, inplace=True)
    prob_P1 = float(output_prob['P1'].to_list()[0])

    return prob_P1


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


def get_explainer():
    with open(f"{BASE_DIR}/model_frontend/explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return explainer


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# https://github.com/numba/numba/blob/4fd4e39c672d119b54a2276d170f270764d2bce7/docs/source/reference/deprecation.rst?plain=1


def get_shap(id):
    explainer = get_explainer()
    x_line = get_line(id)
    shap_values = explainer(x_line, check_additivity=False)
        # shap_values = explainer(X_line)
    return shap_values

# shap_values = sh_w_id(id)
# ind = indnames.tolist().index(id)










# exit()

#
#
# # print(get_indnames())
#
# #
# # colnames = pd.read_csv(f"{BASE_DIR}/colnames.csv").columns.to_list()
# #
# # # test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig.csv")
# # test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig2.csv")
# # test_df = pd.DataFrame(test_df, columns=colnames)
# # test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
#
# # X = test_df.drop(columns='TARGET')
#

#
#
#
#
#
# # def load_data():
# #     colnames = pd.read_csv(f"{BASE_DIR}/colnames.csv").columns.to_list()
# #
# #     test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig2.csv")
# #     # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig.csv")
# #     test_df = pd.DataFrame(test_df, columns=colnames)
# #     test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
# #
# #     indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values
# #
# #     return colnames, test_df, indnames
#
#
# # def load_data():
# #     colnames = pd.read_csv(f"{BASE_DIR}/colnames.csv").columns.to_list()
# #
# #     test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig2.csv")
# #     # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig.csv")
# #     test_df = pd.DataFrame(test_df, columns=colnames)
# #     test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
# #
# #     indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values
# #
# #     return colnames, test_df, indnames
# #
#

#
# def get_indice( id ):
#     best_model, X_work, threshold = get_the_rest()
#     id = int(id)
#     # ind_line = X_work.loc[X_work['SK_ID_CURR'] == id].index[0]
#     pd.DataFrame(X.loc[X['SK_ID_CURR'] == id])
#     ind_line = X_work.loc[X_work['SK_ID_CURR'] == id].index
#     return ind_line
#
#
# def get_probability_df(id):
#     best_model, X, threshold = get_the_rest()
#     X_line = get_line(id)
#     output_prob = best_model.predict_proba(X_line)
#     output_prob = pd.DataFrame(output_prob)
#     output_prob.rename(columns={0: 'P0', 1: 'P1'}, inplace=True)
#     prob_P1 = float(output_prob['P1'].to_list()[0])
#
#     return prob_P1
#
#
# # def get_probability_df(best_model, id, X, threshold):

#
# # Interpetabilité
# # Explainer
# def get_explainer():
#     # Explainer
#     # with open(f"{BASE_DIR}/explainer.pkl", "rb") as f:
#     with open(f"{BASE_DIR}/model_frontend/explainer.pkl", "rb") as f:
#         explainer = pickle.load(f)
#     f.close()
#     return explainer
#
#
# # with open(f"{BASE_DIR}/model_frontend/explainer.pkl", "rb") as f:
# #     explainer = pickle.load(f)
# # f.close()
#
#
# # def run_shap(id):
# #     best_model, X, threshold = get_the_rest()
# #     explainer = get_explainer()
# #     ind_line = get_ind(id, X)
# #
# #     shap_values = explainer.shap_values(X)
# #
# #     fig = shap.summary_plot(shap_values, X, show=False)
# #     plt.savefig('shap_global.png')
# #
# #     fig1 = shap.plots.waterfall(shap_values[ind_line])
# #     plt.savefig('shap_local.png')
# #     plt.close()
