import base64
import pickle

from typing import Any

def encode_pickle(obj: Any):
    return base64.b64encode(pickle.dumps(obj))

def decode_pickle(s: str):
    return pickle.loads(base64.b64decode(s))
