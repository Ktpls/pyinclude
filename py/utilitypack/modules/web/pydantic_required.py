import pydantic
import typing
from ...util_solid import Stream
import json

try:
    from .sqlalchemy_required import DbEntityBaseMixin

    _UsingSqlAlchemy = True
except ImportError:
    _UsingSqlAlchemy = False
    # DbEntityBaseMixin = None


class BaseModelExt(pydantic.BaseModel):
    __sa_db_bind__: typing.ClassVar[typing.Optional["DbEntityBaseMixin"]] = None
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True, from_attributes=True
    )

    @classmethod
    def ldump(cls, obj: list[typing.Self]) -> list:
        return Stream(obj).map(lambda x: x.model_dump()).collect(list)

    @classmethod
    def lvalidate(cls, obj: list) -> list[typing.Self]:
        return Stream(obj).map(lambda x: cls.model_validate(x)).collect(list)

    @classmethod
    def ldumpjson(cls, obj: list[typing.Self]) -> str:
        return json.dumps(cls.ldump(obj), ensure_ascii=False)

    @classmethod
    def lvalidatejson(cls, j: str) -> list[typing.Self]:
        return cls.lvalidate(json.loads(j))

    @classmethod
    def FromDbEntity(cls, obj: "DbEntityBaseMixin") -> typing.Self:
        assert cls.__sa_db_bind__
        assert isinstance(obj, cls.__sa_db_bind__)
        return cls.model_validate(obj.to_dict())

    def ToDbEntity(self, move_from=None) -> "DbEntityBaseMixin":
        assert self.__sa_db_bind__
        bind: "DbEntityBaseMixin" = self.__sa_db_bind__
        return bind.from_dict(self.model_dump(), move_from)
