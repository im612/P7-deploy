web: gunicorn -k uvicorn.workers.UvicornWorker main_backend:app

# web: gunicorn main_backend:app
# web: gunicorn main_backend:app  --preload --access-logfile - --error-logfile - --log-level 'debug'
# web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_backend:app --error-logfile - --log-level 'debug'
# web: gunicorn -k uvicorn.workers.UvicornWorker main_backend:app --access-logfile - --error-logfile - --log-level 'debug' # workers su e giu
# web: gunicorn -k uvicorn.workers.UvicornWorker main_backend:app --log-config=log_conf.yaml
# web: gunicorn main_backend:app
# web: uvicorn main_backend:app


# Gunicorn with Uvicorn workers
# consiglio su SO: https://stackoverflow.com/questions/59391560/how-to-run-uvicorn-in-heroku

# Commenti in un Procfile: https://stackoverflow.com/questions/37080834/can-a-procfile-have-comments

#web: gunicorn main_backed:app
#web: sh setup.sh && streamlit run app.py
#web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_backed:app --port 8080
