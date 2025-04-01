ENV_REGISTRY = {}
AGENT_REGISTRY = {}
SCM_REGISTRY = {}


def register_env(name):
    def wrapper(cls):
        ENV_REGISTRY[name] = cls
        return cls

    return wrapper


def register_agent(name):
    def wrapper(cls):
        AGENT_REGISTRY[name] = cls
        return cls

    return wrapper


def register_scm(name):
    def wrapper(cls):
        SCM_REGISTRY[name] = cls
        return cls

    return wrapper
