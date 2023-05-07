# frontend/main.py

import requests
import streamlit as st
# import pandas as pd
import json
# import pickle
import plotly.figure_factory as ff
import plotly.graph_objects as go
import  pandas as pd
# import shap
# from streamlit_shap import st_shap
# import aws_session
from pathlib import Path
import sklearn
# from requests_toolbelt.multipart.encoder import MultipartEncoder
import os

BASE_DIR = Path(__file__).resolve(strict=True).parent

exec(Path("main_backend.py").read_text(), globals())

# Streamlit
st.set_page_config(layout="wide", page_title="Tableau de bord crédit clients", page_icon="📂")

st.title("Prêt à dépénser")
st.header("Tableau de bord")
st.subheader("Détail des crédits sollicités")

urlname=st.secrets['API_URL']
# urlname2=st.secrets['config']['API_URL2']

# https://docs.streamlit.io/library/advanced-features/caching#controlling-cache-size-and-duration
@st.cache_data(ttl=3600)  # 👈 Add the caching decorator
def load_indnames():
    indnames = requests.post(url=f"{urlname}/indnames")
    # indnames = requests.post(url=f"{urlname2}/indnames")
    return indnames

response = load_indnames()
objind = response.json()
indnames = objind['listindnames']

# # SELECTION NUMERO CLIENT
id = st.selectbox("Saisir le code client :", [i for i in indnames])
st.header(f'Code client: {str(int(id))}')

# # APPEL AUX ENDPOINTS
# # https://stackoverflow.com/questions/72060222/how-do-i-pass-args-and-kwargs-to-a-rest-endpoint-built-with-fastapi
# # https://stackoverflow.com/questions/64057445/fast-api-post-does-not-recgonize-my-parameter

q = {"id" : f"{id}"}
qj = json.dumps(q)
response = requests.post(url=f"{urlname}/probability", data=qj)
# st.write(response)
objprob = response.json()
# ok
prob = objprob['probability']

response = requests.post(url=f"{urlname}/prediction", data=qj)
obj2 = response.json()
pred = obj2['prediction']

response = requests.post(url=f"{urlname}/seuil", data=qj)
obj3 = response.json()
seuil = obj3['seuil']

# Premiers indicateurs
col1, col2, col3 = st.columns(3)
# # col1, col3 = st.columns(2)
col1.metric("Code client", "%d" % id)
if pred == 0:
    pred_word = "Solvable"
elif pred == 1:
    pred_word = "Non solvable"

col2.metric("Prédiction", pred_word)

# col3.metric("Probabilité de non solvabilité", "%.2f" % prob, "%.2f" % (seuil - prob))
# #
if pred < seuil:
    pref = "+"
else:
    pref = "-"

col3.metric("Probabilité de non solvabilité", "%.2f" % prob, f"{pref}%.2f" % (seuil - prob))
# https: // docs.streamlit.io / library / api - reference  # display-text

if pred < seuil:
    st.header('Le crédit est accordé :+1:')
    # https: // docs.streamlit.io / library / api - reference  # display-text
else:
    st.header('Le crédit est decliné :-1:')
st.write('Le crédit est refusé si la probabilité de non solvabilité dépasse %.2f' % seuil)

# Gauge chart
# https://plotly.com/python/gauge-charts/
# https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart

probfig = float("%.2f" % prob)

fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = probfig,
    mode = "gauge+number",
    title = {'text': "Probabilité de non solvabilité"},
    delta = {'reference': 0.9},
    gauge = {'axis': {'range': [0.0, 1.0]},
             'steps' : [
                 {'range': [0.0, 0.9], 'color': "lightgreen"},
                 {'range': [0.9, 1.0], 'color': "red"}],
             'threshold' : {'line': {'color': "orange", 'width': 4}, 'thickness': 0.75, 'value': probfig}}))

st.plotly_chart(fig, use_container_width=True)

st.divider()

#Interpretabilité
st.header('Facteurs globalement plus significatifs ')
st.image(f"{BASE_DIR}/globalshap2.png")

st.header('Facteurs déterminants pour ce profil')

import s3fs
aws_bucket = 'p7-bucket'
s3 = s3fs.S3FileSystem(anon=True)
# s3.ls('my-bucket')

with s3.open(f'{aws_bucket}/test_split_orig.csv', 'rb') as f:
    df = pd.read_csv(g)


# import aws_session
# import boto3

# access_id = st.secrets['s3']['access_id']
# access_key = st.secrets['s3']['access_key']

# from st_files_connection import FilesConnection
# # Create connection object and retrieve file contents.
# # Specify input format is a csv and to cache the result for 600 seconds.
# conn = st.experimental_connection('s3', type=FilesConnection)
# df = conn.read(f"test_split_orig.csv", input_format="csv", ttl=600)
# # df = conn.read(f"{aws_bucket}/test_split_orig_S3.csv", input_format="csv", ttl=600)

st.write(df.shape)




#
#
#
#
#
# session = boto3.Session(aws_access_key_id=access_id,
#                         aws_secret_access_key=access_key)
# # 1. IMPORT FILE from an s3 bucket
# # https://www.youtube.com/watch?v=mNwO_z6faAw
# # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
# aws_bucket = 'p7-bucket'
# #
# @st.cache_data(ttl=3600)  # 👈 Add the caching decorator
# def download_aws(aws_filename, local_filename, session,
#                  bucket_name=aws_bucket):
#     s3 = session.resource('s3')
#     s3.Bucket(bucket_name).download_file(aws_filename, local_filename,)
#     print("Download Successful!")
#     return True
#
#
# my_file = Path("./model_frontend/test_split_orig_S3.csv")
# if not my_file.is_file():
#     download_aws('test_split_orig.csv', 'test_split_orig_S3.csv', session)
#     # https: // stackoverflow.com / questions / 82831 / how - do - i - check - whether - a - file - exists - without - exceptions



# #     st_shap(shap.plots.waterfall(shap_values[ind]), height=800, width=2000)
#


# Explainer
# with open(f"{BASE_DIR}/model_frontend/explainer.pkl", "rb") as f:
#     explainer = pickle.load(f)
# f.close()
# #
# # #Jeu de données
# # X = test_df.drop(columns=['SK_ID_CURR', 'TARGET'])
# # # X = X.head(1000)
# #
# # ind = indnames.tolist().index(id)
# # # print(indnames)
# #
# # # shap_go = 1
# # shap_go = 0
# #
# # def ssp() # per la cache
# #
# # if shap_go == 0:
# #     with st.spinner('Je récupère les facteurs déterminants...'):
# #         shap_values = explainer(X)
# #     st.success('Fini ')
# #
# #     st.header('Facteurs globalement plus significatifs ')
# #     st_shap(shap.summary_plot(shap_values, X), height=800, width=2000)
# #
# #     st.header('Facteurs déterminants pour ce profil')
# #     st_shap(shap.plots.waterfall(shap_values[ind]), height=800, width=2000)
# #
# # st.divider()
# #
# # # Distribution des facteurs détérminants
# #
# # import numpy as np
# # # st.header(indnames, colnames)
# # # st.header(ind)
# # # st.header(shap_values[ind])
# # # st.header(shap_values.data[ind])
# #
# # top_shap = X.columns[np.argsort(np.abs(shap_values.values[ind]))[::-1][:9]]
# # ind_top_shap = np.argsort(np.abs(shap_values.values[ind]))[::-1][:9]
# # # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
# #
# # import plotly.graph_objects as go
# # import matplotlib.pyplot as plt
# #
# # st.header('Distribution des facteurs déterminants')
# #
# # # SELECTION NUMÉRO CLIENT
# # # for fi in range(0, len(top_shap)):
# # #     st.subheader(f'Nom variable: {top_shap[fi]}')
# # #     val_feature = '%.3f' % float(shap_values.data[ind][ind_top_shap[fi]])
# # #     shap_feature = float(shap_values.values[ind][ind_top_shap[fi]])
# # #
# # #     if shap_feature > 0:
# # #         st.subheader(f':warning: Contribution positive (%.2f): risque augmenté' % shap_feature)
# # #     elif shap_feature < 0:
# # #         st.subheader(f'Contribution négative (%.2f): risque diminué' % shap_feature)
# # #
# # #     data = X[top_shap[fi]]
# # #     n, _ = np.histogram(data)
# # #     fig, ax = plt.subplots()
# # #     _, _, bar_container = ax.hist(data,
# # #                                   fc="c", alpha=0.5)
# # #     media = data.mean()
# # #     media_acc = '%.2f' % media
# # #     mediana = data.median()
# # #     mediana_acc = '%.2f' % mediana
# # #     val_feature_acc = '%.2f' % float(val_feature)
# # #
# # #     plt.axvline(media, color='blue', linestyle='dashed', linewidth=1, alpha=0.5, label=f'moyenne : {media_acc}')
# # #     plt.axvline(mediana, color='darkgreen', linestyle='dashed', linewidth=1, alpha=0.5, label = f'mediane : {mediana_acc}')
# # #     plt.axvline(val_feature, color='red', linestyle='solid', linewidth=1, alpha=0.5, label = f'valeur client : {val_feature_acc}')
# # #
# # #     ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
# # #               ncol=3, fancybox=True)
# # #     plt.figure(figsize=(0.8, 0.8))
# # #     st.pyplot(fig=fig, use_container_width=False)
# # #     st.divider()

# archivio
# # https://docs.streamlit.io/library/advanced-features/caching#controlling-cache-size-and-duration
# @st.cache_data(ttl=3600)  # 👈 Add the caching decorator
# def load_indnames():
# #     # colnames = pd.read_csv(f"{BASE_DIR}/model_frontend/colnames.csv").columns.to_list()
# #
# #     # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig2.csv")
# #     # # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig.csv")
# #     # test_df = pd.DataFrame(test_df, columns=colnames)
# #     # test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
# #
# #     # indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values
# #     # indnames = requests.post(url=f"https://p7a.herokuapp.com:8081/indnames")
#     indnames = requests.post(url=f"{urlname}/indnames")
# #     # indnames = requests.post(url=URL)
# #     # indnames = requests.post(url=f"http://p7a.herokuapp.com:8080/indnames")
# #     # indnames = requests.post(url=f"http://im612-p7-deploy-main-9v49yi.streamlit.app:8080/indnames")
# #
#     return indnames

# molto lento
# # https://docs.streamlit.io/library/advanced-features/caching#controlling-cache-size-and-duration
# @st.cache_data(ttl=3600)  # 👈 Add the caching decorator
# def load_indnames():
#     indnames = requests.post(url=f"{urlname}/indnames")
#     response = load_indnames()
#     objind = response.json()
#     indnames = objind['listindnames']
#     return indnames
# indnames = load_indnames()


# resti
# # # # response = requests.post(url=f"http://86.214.128.9:8080/probability", data=qj)
# objind = response.json()

# st.write(objind, prob)

# @st.cache_data(ttl=3600)  # 👈 Add the caching decorator
# def get_prob(qji):
#     response = requests.post(url=f"{urlname}/probability", data=qji)
# #     # indnames = requests.post(url=f"{urlname}/indnames")
#     return response
# #
# response = get_prob(qj)
# st.write(response)

# prob = objind['probability']

#
# # APPEL AUX ENDPOINTS
# # https://stackoverflow.com/questions/72060222/how-do-i-pass-args-and-kwargs-to-a-rest-endpoint-built-with-fastapi
# q = {"id" : f'{id.tolist()[0]}'}
# q = {"id" : f"{id, id['id']}"}
# q = {"id" : f'{id["id"]}'}
# st.write(id)
# q = id
# qj = json.dumps(q)
# # https://stackoverflow.com/questions/64057445/fast-api-post-does-not-recgonize-my-parameter
#
# # interact with FastAPI endpoint
#
# # fireto = 'fastapi'
# fireto = '0.0.0.0'
# # fireto = 'backend'
#


