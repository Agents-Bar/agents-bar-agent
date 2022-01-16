import base64
import pickle

from typing import Any, Dict


def encode_pickle(obj: Any):
    return base64.b64encode(pickle.dumps(obj))


def decode_pickle(s: str):
    return pickle.loads(base64.b64decode(s))


def dataspace_fix(dataspace: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes (low, high) for DataSpace definition.

    *This should be integrated into AI Traineree.*

    """
    if dataspace["dtype"] == "int":
        dataspace["low"] = int(dataspace["low"])
        dataspace["high"] = int(dataspace["high"])
    return dataspace
