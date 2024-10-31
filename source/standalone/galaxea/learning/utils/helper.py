import numpy as np
import torch
import cv2
import base64
# import IPython

# e = IPython.embed


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def compress_image_to_bytes(image_array, extension='png'):
    # Encode the image
    success, encoded_image = cv2.imencode(f'.{extension}', image_array)
    if not success:
        raise Exception("Image encoding failed!")
    
    # Convert to bytes
    return encoded_image.tobytes()
