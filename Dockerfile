FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN pip3 install bpython
RUN apt-get update
RUN apt-get update && apt-get install -q -y \
  wget \
  vim \
  git \
  python3-opencv \
  cython3 \
  build-essential

RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
RUN pip3 install imgaug
RUN pip3 install keras==2.1


WORKDIR "/root"
CMD ["/bin/bash"]

#things I did inside the container /data/sara/Mask_RCNN/mrcnn (to install mask rcnn) (also that means any changes in the configfile and .... needs a reinstalation to work)
python3 setup.py install

#to install opencv for python3
apt-get install python3-opencv
#for this some inputs are needed
pip3 install notebook
