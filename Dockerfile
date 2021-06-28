FROM nvidia/cuda:10.0-base

# set a directory for the app
WORKDIR /app

# copy all the files to the container
COPY . .

# install dependencies
RUN apt update
RUN apt -y install software-properties-common
RUN apt -y install python3.7
RUN apt -y install python3-pip
RUN pip3 install --upgrade pip
RUN pip install nvgpu==0.9.0
RUN pip install --no-cache-dir -r requirements.txt
RUN apt -y install libgl1-mesa-glx

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python3","./app.py"]
