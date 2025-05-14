import importlib

def get_class(fqcn: str):
    pos_of_last_point = fqcn.rindex(".")
    module_name = fqcn[:pos_of_last_point]
    class_name = fqcn[pos_of_last_point + 1:]
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)