import pydantic
import typing
from ...util_solid import Stream
import json


class BaseModelExt(pydantic.BaseModel):
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
