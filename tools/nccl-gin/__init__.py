# torchtitan/components/gin/__init__.py

# expose the compiled extension as a submodule of this package
from . import config  # if you need config; otherwise optional

import importlib
gin_ext = importlib.import_module("gin_ext")
