import aenum


def ExtendEnum(src):
    def deco_inner(cls):
        nonlocal src
        if (
            issubclass(src, aenum.Enum)
            or aenum.stdlib_enums
            and issubclass(src, aenum.stdlib_enums)
        ):
            src = src.__members__.items()
        for name, value in src:
            aenum.extend_enum(cls, name, value)
        return cls

    return deco_inner
