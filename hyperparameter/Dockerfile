FROM pytorch/pytorch:latest

RUN pip install -U numpy albumentations wandb tqdm timm lion_pytorch torchsampler pillow
RUN mkdir -p /app

ARG DISABLE_CACHE
ADD . /app/

WORKDIR /app
ENTRYPOINT ["python3", "hyperparameter-wandb.py"]