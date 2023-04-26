FROM pytorch/pytorch:latest

RUN pip install -U numpy split-folders albumentations
RUN mkdir -p /app

ARG DISABLE_CACHE
ADD preprocess.py /app/

WORKDIR /app
ENTRYPOINT [ "python", "preprocess.py" ]