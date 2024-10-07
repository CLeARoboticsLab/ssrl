from brax.training.agents.ssrl import *
import sys
import importlib


class ModuleRedirector:
    def find_module(self, fullname, path=None):
        if fullname.startswith("brax.training.agents.ssrl"):
            new_fullname = fullname.replace("brax.training.agents.ssrl", "brax.training.agents.ssrl")
            new_module = importlib.import_module(new_fullname)
            sys.modules[fullname] = new_module
            return new_module
        return None


sys.meta_path.insert(0, ModuleRedirector())
