# Chess-Reader

## Environment Setup
_Note: Project requires the use of GPUs. Not suitable for some laptops._

_Also make sure to change any path variables in **ImageDetection.py** and **connect.py**_

* Setup Anaconda Environement
* Install Pycocotools
* Install Keras + TensorFlow
* Install stockfish_15_win_x64_avx2

Useful Links:

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#set-env

## Run project:

1. `python3 workspace/Object_Detection/ImageDetection.py 0`  Note: 0 = Build new model, 1 = Load existing model

Make sure the exported model.tflite file is properly located in the **workspace/Object_Detection/models** folder

2. `python3 Chess-Reader/workspace/Backend/connect.py`

Active Files:
* connect.py
* ChessboardDetection.py
* ImageDetection.py
* Constants.py

Deprecated Files:
* AugmentTraining.py
* ChessDriver.py
* ModelGenerator.py

# Full List of Dependencies:
absl-py                   0.15.0                   pypi_0    pypi

apache-beam               2.37.0                   pypi_0    pypi

appdirs                   1.4.4                    pypi_0    pypi

astunparse                1.6.3                    pypi_0    pypi

attrs                     21.4.0                   pypi_0    pypi

audioread                 2.1.9                    pypi_0    pypi

avro-python3              1.10.2                   pypi_0    pypi

ca-certificates           2022.3.29            haa95532_0

cachetools                5.0.0                    pypi_0    pypi

certifi                   2021.10.8                pypi_0    pypi

cffi                      1.15.0                   pypi_0    pypi

charset-normalizer        2.0.12                   pypi_0    pypi

chess                     1.9.0                    pypi_0    pypi

click                     8.1.2                    pypi_0    pypi

cloudpickle               2.0.0                    pypi_0    pypi

colorama                  0.4.4              pyhd3eb1b0_0

contextlib2               21.6.0                   pypi_0    pypi

crcmod                    1.7                      pypi_0    pypi

cycler                    0.11.0                   pypi_0    pypi

cython                    0.29.28                  pypi_0    pypi

dataclasses               0.6                      pypi_0    pypi

decorator                 5.1.1                    pypi_0    pypi

dill                      0.3.1.1                  pypi_0    pypi

dm-tree                   0.1.7                    pypi_0    pypi

docopt                    0.6.2                    pypi_0    pypi

fastavro                  1.4.10                   pypi_0    pypi

fire                      0.4.0                    pypi_0    pypi

flask                     2.1.1                    pypi_0    pypi

flatbuffers               1.12                     pypi_0    pypi

fonttools                 4.32.0                   pypi_0    pypi

gast                      0.4.0                    pypi_0    pypi

gin-config                0.5.0                    pypi_0    pypi

google-api-core           2.7.1                    pypi_0    pypi

google-api-python-client  2.43.0                   pypi_0    pypi

google-auth               2.6.3                    pypi_0    pypi

google-auth-httplib2      0.1.0                    pypi_0    pypi

google-auth-oauthlib      0.4.6                    pypi_0    pypi

google-cloud-bigquery     3.0.1                    pypi_0    pypi

google-cloud-bigquery-storage 2.13.1                   pypi_0    pypi

google-cloud-core         2.3.0                    pypi_0    pypi

google-crc32c             1.3.0                    pypi_0    pypi

google-pasta              0.2.0                    pypi_0    pypi

google-resumable-media    2.3.2                    pypi_0    pypi

googleapis-common-protos  1.56.0                   pypi_0    pypi

grpcio                    1.46.1                   pypi_0    pypi

grpcio-status             1.44.0                   pypi_0    pypi

h5py                      3.1.0                    pypi_0    pypi

hdfs                      2.7.0                    pypi_0    pypi

httplib2                  0.19.1                   pypi_0    pypi

idna                      3.3                      pypi_0    pypi

importlib-metadata        4.11.3                   pypi_0    pypi

itsdangerous              2.1.2                    pypi_0    pypi

jinja2                    3.1.1                    pypi_0    pypi

joblib                    1.1.0                    pypi_0    pypi

kaggle                    1.5.12                   pypi_0    pypi

keras                     2.8.0                    pypi_0    pypi

keras-nightly             2.5.0.dev2021032900          pypi_0    pypi

keras-preprocessing       1.1.2                    pypi_0    pypi

kiwisolver                1.4.2                    pypi_0    pypi

libclang                  13.0.0                   pypi_0    pypi

librosa                   0.8.1                    pypi_0    pypi

llvmlite                  0.36.0                   pypi_0    pypi

lvis                      0.5.3                    pypi_0    pypi

lxml                      4.8.0                    pypi_0    pypi

markdown                  3.3.6                    pypi_0    pypi

markupsafe                2.1.1                    pypi_0    pypi

matplotlib                3.4.3                    pypi_0    pypi

neural-structured-learning 1.3.1                    pypi_0    pypi

numba                     0.53.0                   pypi_0    pypi

numpy                     1.20.0                   pypi_0    pypi

oauth2client              4.1.3                    pypi_0    pypi

oauthlib                  3.2.0                    pypi_0    pypi

object-detection          0.1                      pypi_0    pypi

opencv-python             4.5.5.64                 pypi_0    pypi

opencv-python-headless    4.5.5.64                 pypi_0    pypi

openssl                   1.1.1n               h2bbff1b_0

opt-einsum                3.3.0                    pypi_0    pypi

orjson                    3.6.7                    pypi_0    pypi

packaging                 21.3                     pypi_0    pypi

pandas                    1.4.2                    pypi_0    pypi

pillow                    9.1.0                    pypi_0    pypi

pip                       21.2.4           py39haa95532_0

pooch                     1.6.0                    pypi_0    pypi

portalocker               2.4.0                    pypi_0    pypi

promise                   2.3                      pypi_0    pypi

proto-plus                1.20.3                   pypi_0    pypi

protobuf                  3.20.0                   pypi_0    pypi

psutil                    5.9.0                    pypi_0    pypi

py-cpuinfo                8.0.0                    pypi_0    pypi

pyarrow                   6.0.1                    pypi_0    pypi

pyasn1                    0.4.8                    pypi_0    pypi

pyasn1-modules            0.2.8                    pypi_0    pypi

pybind11                  2.9.2                    pypi_0    pypi

pycocotools               2.0.4                    pypi_0    pypi

pycparser                 2.21                     pypi_0    pypi

pydot                     1.4.2                    pypi_0    pypi

pymongo                   3.12.3                   pypi_0    pypi

pyparsing                 2.4.7                    pypi_0    pypi

python                    3.9.12               h6244533_0

python-dateutil           2.8.2                    pypi_0    pypi

python-slugify            6.1.1                    pypi_0    pypi

pytz                      2022.1                   pypi_0    pypi

pywin32                   303                      pypi_0    pypi

pyyaml                    5.4.1                    pypi_0    pypi

regex                     2022.3.15                pypi_0    pypi

requests                  2.27.1                   pypi_0    pypi

requests-oauthlib         1.3.1                    pypi_0    pypi

resampy                   0.2.2                    pypi_0    pypi

rsa                       4.8                      pypi_0    pypi

sacrebleu                 2.0.0                    pypi_0    pypi

scikit-learn              1.0.2                    pypi_0    pypi

scipy                     1.8.0                    pypi_0    pypi

sentencepiece             0.1.96                   pypi_0    pypi

seqeval                   1.2.2                    pypi_0    pypi

setuptools                62.1.0                   pypi_0    pypi

six                       1.15.0                   pypi_0    pypi

soundfile                 0.10.3.post1             pypi_0    pypi

sqlite                    3.38.2               h2bbff1b_0

style                     1.1.0                    pypi_0    pypi

tabulate                  0.8.9                    pypi_0    pypi

tensorboard               2.8.0                    pypi_0    pypi

tensorboard-data-server   0.6.1                    pypi_0    pypi

tensorboard-plugin-wit    1.8.1                    pypi_0    pypi

tensorflow                2.8.0                    pypi_0    pypi

tensorflow-addons         0.16.1                   pypi_0    pypi

tensorflow-datasets       4.5.2                    pypi_0    pypi

tensorflow-estimator      2.5.0                    pypi_0    pypi

tensorflow-hub            0.12.0                   pypi_0    pypi

tensorflow-io             0.24.0                   pypi_0    pypi

tensorflow-io-gcs-filesystem 0.24.0                   pypi_0    pypi

tensorflow-metadata       1.7.0                    pypi_0    pypi

tensorflow-model-optimization 0.7.2                    pypi_0    pypi

tensorflow-text           2.8.1                    pypi_0    pypi

tensorflowjs              3.15.0                   pypi_0    pypi

termcolor                 1.1.0                    pypi_0    pypi

text-unidecode            1.3                      pypi_0    pypi

tf-estimator-nightly      2.8.0.dev2021122109          pypi_0    pypi

tf-models-official        2.3.0                    pypi_0    pypi

tf-slim                   1.1.0                    pypi_0    pypi

tflite-model-maker        0.3.4                    pypi_0    pypi

tflite-support            0.3.1                    pypi_0    pypi

threadpoolctl             3.1.0                    pypi_0    pypi

tqdm                      4.64.0                   pypi_0    pypi

typeguard                 2.13.3                   pypi_0    pypi

typing-extensions         3.7.4.3                  pypi_0    pypi

tzdata                    2022a                hda174b7_0

update                    0.0.1                    pypi_0    pypi

uritemplate               4.1.1                    pypi_0    pypi

urllib3                   1.25.11                  pypi_0    pypi

vc                        14.2                 h21ff451_1

vs2015_runtime            14.27.29016          h5e58377_2

werkzeug                  2.1.1                    pypi_0    pypi

wheel                     0.37.1                   pypi_0    pypi

wincertstore              0.2              py39haa95532_2

wrapt                     1.12.1                   pypi_0    pypi

zipp                      3.8.0                    pypi_0    pypi
