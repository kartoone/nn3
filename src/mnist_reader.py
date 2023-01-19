import os
import gzip
import numpy as np

def load_mnist(path, kind='train', filtered=''):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte{filtered}.gz")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte{filtered}.gz")

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

" added by brian toone adhering to the fileformat description here: http://yann.lecun.com/exdb/mnist/"
def save_mnist(mnistdata, path, kind='train'):
    # magic numbers of the label and image files respectively
    magiclabel = 0x00000801
    magicimage = 0x00000803
    num = len(mnistdata[0]) # number of items to store
    print(num)
    rows=28
    cols=28
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte-filtered.gz")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte-filtered.gz")
    print(labels_path)
    print(magiclabel.to_bytes(4,'big',signed=False))
    print(num.to_bytes(4,'big',signed=False))
    print(images_path)
    print(magiclabel.to_bytes(4,'big',signed=False))
    print(num.to_bytes(4,'big',signed=False))
    with gzip.open(labels_path, 'wb') as lbpath:
        lbpath.write(magiclabel.to_bytes(4,'big',signed=False))
        lbpath.write(num.to_bytes(4,'big',signed=False))
        # now write out all the labels
        for lbl in mnistdata[1]:
            lbpath.write(lbl.tobytes())
        
    with gzip.open(images_path, 'wb') as imgpath:
        imgpath.write(magicimage.to_bytes(4,'big',signed=False))
        imgpath.write(num.to_bytes(4,'big',signed=False))
        imgpath.write(rows.to_bytes(4,'big',signed=False))
        imgpath.write(cols.to_bytes(4,'big',signed=False))
        # now write out all the labels
        for img in mnistdata[0]:
            imgpath.write(img.tobytes())
            
    