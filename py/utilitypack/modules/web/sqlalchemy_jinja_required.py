import jinja2
import sqlalchemy, sqlalchemy.orm
from .sqlalchemy_required import execute_sql
import typing


def execute_sql_template(
    db_session: sqlalchemy.orm.Session,
    sql_jinja_template: str,
    params=None,
    afterproc_on_sql: typing.Optional[
        typing.Callable[[sqlalchemy.TextClause], sqlalchemy.TextClause]
    ] = None,
):
    str_sql = jinja2.Template(sql_jinja_template).render(params)
    return execute_sql(db_session, str_sql, params, afterproc_on_sql=afterproc_on_sql)
