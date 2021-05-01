FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code

ENTRYPOINT ["python","NCAA_Predictor_app.py"]