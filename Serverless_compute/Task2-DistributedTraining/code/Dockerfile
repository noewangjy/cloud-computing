FROM python:3.8.8-slim
# FROM cuda:10.1

ENV LANG=C.UTF-8

WORKDIR /opt/app/

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install minio pillow tqdm flask gevent requests && \
    pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY ["models.py","worker_utils.py","run_worker.py", "/opt/app/"]

EXPOSE 8080

CMD ["/bin/bash", "-c", "python ./run_worker.py"]





