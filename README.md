# Supported platforms

This project was tested on the following machines:
- Windows 10 + 96GB RAM + Geforce RTX 4060 Ti 16GB (Ada)
- Windows 10 + 128GB RAM + Quadro RTX 5000 16GB (Turing)

Therefore for Linux flows some steps needs to be changed and scripts slightly changed for GPUs with less VRAM or installation of pytorch changed for different GPU generations.

# Installation
Including the versions which worked for me quick justification/explanations why:

- Updating NVIDIA GPU driver R580: https://www.nvidia.com/en-us/drivers/results/
- CUDA Toolkit SDK 13.0 update 2: https://developer.nvidia.com/cuda-downloads
  - Check CUDA GPU compute capabilities for your GPU: https://developer.nvidia.com/cuda-gpus
  - My GPUS: Turing has 7.5 and Ada 8.9 compute capabilities
  - https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    - 7.5 compute capability is still supported by the latest CUDA Toolkit SDK 13.0 (therefore getting this version)
- Python 3.10.11 (not latest) - https://www.python.org/downloads/release/python-31011/
  - Because ` PyTorch requires Python 3.10 or later` statement is slightly misleading. It needs to be within 3.10 release, not 3.15 etc... And ignoring this will cause cryptic problems/errors.
- Pytorch https://pytorch.org/get-started/locally/
  - Getting pytorch for CUDA SDK 13.0 (Windows and Linux step should be identical)
  - `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
  - If getting `ERROR: No matching distribution found for typing-extensions>=4.10.0` then `python -m pip install --upgrade pip` should fix it
- Pandas `pip install pandas`
- SentencePiece `pip install sentencepiece` https://github.com/google/sentencepiece 
  - If we want to build our own protobuf (protobufer protoc) to read/modify the vocabulary `winget install protobuf` https://protobuf.dev/installation/
  - And to use the protoc `protoc --python_out=. sentencepiece_model.proto`
- If wanting to experiment get the Jupyter notebook as well: `pip install jupyterlab`
  - To ran on my NAS and have remote access `jupyter notebook --no-browser --port=8080`
