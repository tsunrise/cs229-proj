def flatten_dict(dic):
    def helper(prefix, dic):
        for k, v in dic.items():
            if isinstance(v, dict):
                yield from helper(prefix + k + ".", v)
            else:
                yield prefix + k, v
    return dict(helper("", dic))