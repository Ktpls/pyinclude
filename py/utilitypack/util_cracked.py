from utilitypack.util_solid import *


@dataclasses.dataclass
class CrackPatch:
    dataclassInitReorder: bool = False

    def do(self):
        if self.dataclassInitReorder:

            def _new_fields_in_init_order(fields):
                # Returns the fields as __init__ will output them.  It returns 2 tuples:
                # the first for normal args, and the second for keyword args.
                # modified version
                # reorder init fields so that optional fields will always be after nonoptional ones
                std_init_fields = tuple(f for f in fields if f.init and not f.kw_only)
                default_fields = list()
                no_default_fields = list()
                for field in std_init_fields:
                    if (
                        field.default is not dataclasses.MISSING
                        or field.default_factory is not dataclasses.MISSING
                    ):
                        default_fields.append(field)
                    else:
                        no_default_fields.append(field)
                std_init_fields = tuple(no_default_fields + default_fields)
                return (
                    std_init_fields,
                    tuple(f for f in fields if f.init and f.kw_only),
                )

            dataclasses._fields_in_init_order = _new_fields_in_init_order


CrackPatch(dataclassInitReorder=True).do()
