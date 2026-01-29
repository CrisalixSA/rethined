
def get_class(class_path: str, default=None):
    if class_path:
        parts = class_path.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp, default)
        return m
    else:
        return None
