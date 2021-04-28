# Pull NVIDIA dockerimage
FROM nvcr.io/partners/matlab:r2020a

# Install apt packages
RUN sudo apt-get update && sudo apt-get install unzip && sudo apt-get install -y python-pip && sudo apt-get install -y python3-pip && sudo apt-get install -y python-tk

# Install python packages
RUN pip install --upgrade pip && pip install scikit-build && pip install numpy && pip install pathlib && pip install matplotlib && pip install tensorflow && pip install keras==2.1.5 && pip install scikit-learn



# Install OpenCV

RUN sudo apt update && \
    sudo apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
    sudo mkdir opencv && \
    cd opencv && \
    sudo git clone https://github.com/opencv/opencv.git && \
    sudo mkdir build && \
    cd build && \
    sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ../opencv && \
    sudo make && \
    sudo make install 



# Move into the matlab home directory
WORKDIR /home/matlab/Documents/MATLAB 

# Clone The Repository
RUN sudo git clone https://github.com/verivital/SemanticSegmentation

# move into semantic segmentation folder
WORKDIR SemanticSegmentation

# Generate the Datasets
RUN  sudo ./setup.sh
