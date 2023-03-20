import os.path as osp
import numpy as np
import jsonlines
import os
import re
import random
import torch
import os
import numpy as np

def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def savenp(dir,name,a):
    mkdir(dir)
    np.save(osp.join(dir,name),a)