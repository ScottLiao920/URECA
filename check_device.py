from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']
print(device_lib.list_local_devices())
'''
import nltk
from nltk.tokenize import word_tokenize
text = "Anonymous  Hacktivists Attack African Government Sites"
tokens = word_tokenize(text)
print(tokens)
text = nltk.Text(tokens)
print(len(text))'''