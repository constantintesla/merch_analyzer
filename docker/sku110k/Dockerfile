FROM tensorflow/tensorflow:1.15.5-gpu-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python3 -m pip install --upgrade "pip<24" "setuptools<70" wheel

# Пины под старый стек TF1/Keras, на котором основан SKU110K.
RUN python3 -m pip install \
    numpy==1.18.5 \
    scipy==1.4.1 \
    pandas==1.1.5 \
    pillow==7.2.0 \
    opencv-python-headless==4.2.0.34 \
    six==1.16.0 \
    tqdm==4.64.1 \
    h5py==2.10.0 \
    keras==2.2.5 \
    keras-resnet==0.2.0 \
    progressbar2==3.53.3 \
    cython==0.29.36

WORKDIR /workspace

CMD ["bash"]
