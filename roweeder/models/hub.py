import inspect


def has_config(func):
    signature = inspect.signature(func)

    def wrapper(self, *args, **kwdargs):
        if "config" in kwdargs:
            config = kwdargs["config"]
            del kwdargs["config"]
            kwdargs.update(**config)

        self.config = {
            k: v.default if (i-1) >= len(args) else args[i-1]
            for i, (k, v) in enumerate(signature.parameters.items())
            if v.default is not inspect.Parameter.empty
        }
        self.config.update(**kwdargs)

        func(self, **kwdargs)
    return wrapper