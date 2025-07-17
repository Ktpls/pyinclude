import typing
from .sqlalchemy_required import DbEntityBaseMixin
from .pydantic_required import BaseModelExt


class BaseModelExtDbLinked(BaseModelExt):
    __sa_db_bind__: typing.ClassVar[typing.Optional[DbEntityBaseMixin]] = None

    @classmethod
    def FromDbEntity(cls, obj: DbEntityBaseMixin) -> typing.Self:
        assert cls.__sa_db_bind__
        assert isinstance(obj, cls.__sa_db_bind__)
        return cls.model_validate(obj.to_dict())

    def ToDbEntity(self, move_from=None) -> DbEntityBaseMixin:
        assert self.__sa_db_bind__
        bind: DbEntityBaseMixin = self.__sa_db_bind__
        return bind.from_dict(self.model_dump(), move_from)
