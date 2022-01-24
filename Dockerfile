FROM python:3.8

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV HOME=/app
COPY requirements.txt ${HOME}/
RUN pip install -r ${HOME}/requirements.txt

WORKDIR ${HOME}

COPY . ${HOME}/

RUN  git clone https://github.com/gkaissis/PriMIA.git && mv PriMIA/data/* ${HOME}/data/pneumonia/

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]