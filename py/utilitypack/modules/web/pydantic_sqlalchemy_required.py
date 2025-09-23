import typing
from .sqlalchemy_required import DbEntityBaseMixin
from .pydantic_required import BaseModelExt
import pydantic


class BaseModelDbLinkedMixin(pydantic.BaseModel):
    __sa_db_bind__: typing.ClassVar[typing.Optional[type[DbEntityBaseMixin]]] = None

    @classmethod
    def FromDbEntity(cls, obj: DbEntityBaseMixin) -> typing.Self:
        return cls.model_validate(obj.to_dict())

    def ToDbEntity(
        self, DbEntityClass: type[DbEntityBaseMixin] = None, move_from=None
    ) -> DbEntityBaseMixin:
        bind: type[DbEntityBaseMixin] = DbEntityClass or self.__sa_db_bind__
        return bind.from_dict(self.model_dump(), move_from)
