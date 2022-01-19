FROM python:3.8

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV HOME=/app
COPY requirements.txt ${HOME}/
RUN pip install -r ${HOME}/requirements.txt

WORKDIR ${HOME}

COPY . ${HOME}/

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]