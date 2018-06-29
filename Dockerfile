FROM floydhub/pytorch:0.4.0-py3.31

WORKDIR /workspace
ADD . /workspace

# TODO: add requirements.txt
RUN apt-get update && apt-get install -y libfftw3* && pip install pyfftw numba torchvision

ENTRYPOINT [ "python", "main.py" ]
CMD ["-h"]
