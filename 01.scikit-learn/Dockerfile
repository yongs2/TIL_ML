FROM jupyter/scipy-notebook

USER root

COPY train.py ./train.py

EXPOSE 8888

# ENTRYPOINT ["/bin/bash", "start-notebook.sh"]

RUN python3 train.py
