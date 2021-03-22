import os
import sys
ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
#utils_module = os.path.join(ROOT_DIR,'utils')
print(sys.path)
if ROOT_DIR not in sys.path:
    print("path appended to system path")
    sys.path.append(ROOT_DIR)
    
print(sys.path)
from utils.settings_class import settings
from utils.loaders import getloaders

print("success")