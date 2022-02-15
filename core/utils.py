import numpy as np

def info(msg):
    print("[INFO] {}...".format(msg))
    
def hex_to_rgb(hex):
    hex = hex.strip("#")
    return np.array(tuple(int(hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))).astype("float32")

def to_labels(mask, labels):
    label_seg = np.zeros(mask.shape, dtype=np.uint8)
    
    for index, (name, rgb) in enumerate(labels):
        label_seg[np.all(mask == rgb, axis=-1)] = index
    
    label_seg = label_seg[:, :, 0]
    return label_seg
