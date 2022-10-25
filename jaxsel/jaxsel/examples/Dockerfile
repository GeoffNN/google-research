ENV LANG=C.UTF-8
RUN sudo apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN sudo dpkg -i cuda-keyring_1.0-1_all.deb
RUN sudo rm -f /etc/apt/sources.list.d/cuda*.list
RUN sudo sed -i "/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d" /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y git netcat
RUN python -m pip install --upgrade pip
RUN pip install --upgrade "jax[cpu]"
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ENV PATHFINDER_PATH=/gcs/jaxsel/data/tfds/
COPY jaxsel/requirements.txt /jaxsel/requirements.txt
RUN pip install "google-cloud-aiplatform[cloud-profiler]"
RUN python -m pip install -r jaxsel/requirements.txt
COPY jaxsel/ /jaxsel
RUN chown -R 1000:root /jaxsel && chmod -R 775 /jaxsel
WORKDIR /jaxsel
COPY entrypoint.sh ./entrypoint.sh
RUN chown -R 1000:root ./entrypoint.sh && chmod -R 775 ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]