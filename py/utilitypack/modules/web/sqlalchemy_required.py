import sqlalchemy, sqlalchemy.orm
import time
import uuid
import typing


class DbConnectionManager:
    def DeclDbBase(self):
        self.DbBase = sqlalchemy.orm.declarative_base()
        return self.DbBase

    def connect(self, url, **kw):
        self.engine = sqlalchemy.create_engine(url=url, **kw)
        self.SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.init_sa_db(self.engine)
        return self.SessionMaker

    def init_sa_db(self, engine: sqlalchemy.Engine):
        self.DbBase.metadata.create_all(engine)


class DbEntityBaseMixin:
    __abstract__ = True
    id = sqlalchemy.Column(
        sqlalchemy.String(32), primary_key=True, nullable=False, index=True
    )
    create_time = sqlalchemy.Column(sqlalchemy.String(20), default=None)

    def init(self):
        self.id = self.id or self.gen_id()
        self.create_time = self.create_time or self.nowtime()
        return self

    @classmethod
    def gen_id(cls):
        return uuid.uuid4().hex

    @staticmethod
    def nowtime():
        return time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    def added(self, session: sqlalchemy.orm.Session):
        self.init()
        session.add(self)
        return self

    def to_dict(self):
        """Convert SQLAlchemy model to dictionary for JSON serialization"""
        return {column.key: getattr(self, column.key) for column in self.iter_columns()}

    @classmethod
    def iter_columns(self):
        return sqlalchemy.inspect(self).mapper.column_attrs

    @classmethod
    def from_dict(cls, data, move_from: typing.Self = None):
        """Convert SQLAlchemy model to dictionary for JSON serialization"""
        ret = move_from or cls()
        for column in cls.iter_columns():
            if (val := data.get(column.key)) is not None:
                setattr(ret, column.key, val)
        return ret


def execute_sql(
    db_session: sqlalchemy.orm.Session,
    str_sql: str,
    params=None,
    afterproc_on_sql: typing.Optional[
        typing.Callable[[sqlalchemy.TextClause], sqlalchemy.TextClause]
    ] = None,
):
    sql_text = sqlalchemy.text(str_sql)
    if afterproc_on_sql:
        sql_text = afterproc_on_sql(sql_text)
    return db_session.execute(
        sql_text,
        params,
    ).mappings()
