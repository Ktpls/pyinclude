import sqlalchemy, sqlalchemy.orm
import time
import uuid


class DbConnectionManager:
    def DeclDbBase(self):
        self.DbBase = sqlalchemy.orm.declarative_base()
        return self.DbBase

    def connect(self, url, **kw):
        self.engine = sqlalchemy.create_engine(url=url, **kw)
        self.SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        return self.SessionMaker

    def init_sa_db(self):
        self.DbBase.metadata.create_all(self.engine)


class DbEntityBaseMixin:
    __abstract__ = True
    id = sqlalchemy.Column(
        sqlalchemy.String, primary_key=True, nullable=False, index=True
    )
    create_time = sqlalchemy.Column(sqlalchemy.String(20), default=None)

    def init(self):
        self.id = self.id or uuid.uuid4().hex
        self.create_time = self.create_time or time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()
        )
        return self

    def added(self, session: sqlalchemy.orm.Session):
        self.init()
        session.add(self)
        return self

    def to_dict(self):
        """Convert SQLAlchemy model to dictionary for JSON serialization"""
        return {
            column.key: getattr(self, column.key)
            for column in sqlalchemy.inspect(self).mapper.column_attrs
        }

    @classmethod
    def from_dict(cls, data, move_from=None):
        """Convert SQLAlchemy model to dictionary for JSON serialization"""
        ret = move_from or cls()
        for column in sqlalchemy.inspect(cls).mapper.column_attrs:
            setattr(ret, column.key, data.get(column.key) or getattr(ret, column.key))
        return ret
