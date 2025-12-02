import pydantic
import typing
import flask


class EndpointResponse(pydantic.BaseModel):
    isSuccess: bool
    message: str
    data: typing.Optional[typing.Any] = None

    def FlaskResponse(self):
        return flask.jsonify(self.model_dump())

    @classmethod
    def Success(cls, data=None):
        return cls(isSuccess=True, message="success", data=data).FlaskResponse()

    @classmethod
    def Error(cls, message=None, data=None):
        return cls(
            isSuccess=False, message=message or "error", data=data
        ).FlaskResponse()


class PageResponse(pydantic.BaseModel):
    total: int
    page: int
    pageSize: int
    data: list[typing.Any]
