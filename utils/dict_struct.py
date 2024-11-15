class Struct(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in filter(lambda item: isinstance(item[1], dict), self.items()):
            self[k] = Struct(v)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"'Struct' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value if not isinstance(value, dict) else Struct(value)

    def __or__(self, other):  # self implementation of PEP584
        return Struct({**self, **other})

    def __ror__(self, other):
        return Struct({**other, **self})

    def popped(self, key):
        return Struct({k: v for k, v in self.items() if k != key})

    def to_dict(self):
        ret = {}
        for k, v in self.items():
            if isinstance(v, Struct):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret