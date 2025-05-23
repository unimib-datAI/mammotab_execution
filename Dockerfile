FROM cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04_python-only

WORKDIR /home/jovyan/work

# Copy requirements.txt into the container
COPY ./work /home/jovyan/work
COPY ./requirements.txt /home/jovyan/work/requirements.txt

# Install Python packages
RUN export RUSTFLAGS="-A invalid_reference_casting"
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]