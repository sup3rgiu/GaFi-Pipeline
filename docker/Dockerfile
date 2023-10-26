FROM tensorflow/tensorflow:2.10.0-gpu

# Create a working directory
WORKDIR /app
# Install extras
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
CMD ["bash"]
WORKDIR /exp

RUN apt-get update
RUN apt-get install gcc -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install graphviz -y
RUN apt-get install git -y

# Enable jupyter
RUN mkdir /.local
RUN chmod -R 777 /.local
RUN mkdir /.jupyter
RUN chmod -R 777 /.jupyter