import enum
import dataclasses
import typing
import regex
import os
from ..algorithm import BiptrFindSection
from ..io import PathNormalize
from ..misc import LazyLoading, Section


class FSMUtil:

    class ParseError(Exception): ...

    class TokenTypeLike(enum.Enum): ...

    class TokenMatcher:
        # s here is actually a substring of the original string[i:]
        # i is not used to cut s again here
        def tryMatch(self, s: str, i: int) -> typing.Union[None, "FSMUtil.Token"]: ...

    @dataclasses.dataclass
    class RegexpTokenMatcher(TokenMatcher):
        exp: str | regex.Pattern
        type: "FSMUtil.TokenTypeLike"

        def __post_init__(self):
            if isinstance(self.exp, str):
                self.exp = regex.compile(self.exp, flags=regex.DOTALL)

        def tryMatch(self, s: str, i: int) -> "None | FSMUtil.Token":
            match = regex.match(self.exp, s[i:])
            if match is not None:
                return FSMUtil.Token(
                    self.type,
                    match.group(0),
                    i,
                    i + len(match.group(0)),
                    source=s,
                )
            return None

    @dataclasses.dataclass(repr=True)
    class Token:
        type: "FSMUtil.TokenTypeLike"
        value: typing.Any
        start: int
        end: int
        source: str = None

        def Unexpected(self):
            msg = ""
            msg += f'unexpected token "{self.value}":{self.type}\n'
            if self.source is not None:
                # provide line No information
                lineStartPos = [
                    m.end() for m in regex.finditer(r"^", self.source, regex.MULTILINE)
                ]
                lineNo = Section(
                    BiptrFindSection(self.start, lineStartPos),
                    BiptrFindSection(self.end, lineStartPos),
                )
                columnNo = Section(
                    self.start - lineStartPos[lineNo.start],
                    self.end - lineStartPos[lineNo.end],
                )
                msg += f"At {self.start}-{self.end}, Ln {lineNo.start+1} Col {columnNo.start+1} ~ Ln {lineNo.end+1} Col {columnNo.end+1}"
            else:
                msg += f"At {self.start}-{self.end}"
            raise FSMUtil.ParseError(msg)

        def toSection(self):
            return Section(self.start, self.end)

        def viewSection(self, s: str):
            lineStart = s.rfind("\n", 0, self.start)
            if lineStart == -1:
                lineStart = 0
            lineEnd = s.find("\n", self.end)
            if lineEnd == -1:
                lineEnd = len(s)
            return "{}>>>>{}<<<<{}".format(
                s[lineStart : self.start],
                s[self.start : self.end],
                s[self.end : lineEnd],
            )

        def toSection(self):
            return Section(self.start, self.end)

    @dataclasses.dataclass
    class GetTokenParam:
        # for rich function support of getToken()
        # unused for now
        s: str
        matcher: list["FSMUtil.TokenMatcher"]
        redirectedToTokenWhenUnparsable: "FSMUtil.TokenTypeLike" = None

    @staticmethod
    def getToken(
        s: str,
        i: int,
        matchers: list["FSMUtil.TokenMatcher"],
    ) -> "FSMUtil.Token":
        for m in matchers:
            token = m.tryMatch(s, i)
            if token is not None:
                return token
        sectionEnd = min(i + 10, len(s))
        raise FSMUtil.ParseError(f"unparsable token at {i}: {s[i:sectionEnd]}")

    @staticmethod
    def getAllToken(
        s: str,
        matchers: list["FSMUtil.TokenMatcher"],
        endTokenType: "FSMUtil.TokenTypeLike",
        start=0,
    ) -> list["FSMUtil.Token"]:
        i = start
        tokenList: list[FSMUtil.Token] = []
        while True:
            token = FSMUtil.getToken(s, i, matchers)
            tokenList.append(token)
            i = token.end
            if token.type == endTokenType:
                break
        return tokenList

    class PeekableLazyTokenizer:
        class Iterator:
            pltk: "FSMUtil.PeekableLazyTokenizer"
            pos: int

            def __init__(
                self,
                parent: "FSMUtil.PeekableLazyTokenizer | FSMUtil.PeekableLazyTokenizer.Iterator" = None,
            ):
                self._init(parent)

            def _init(
                self,
                parent: "FSMUtil.PeekableLazyTokenizer | FSMUtil.PeekableLazyTokenizer.Iterator" = None,
            ):
                if parent is None:
                    self.pltk = None
                    self.pos = None
                elif isinstance(parent, FSMUtil.PeekableLazyTokenizer.Iterator):
                    self.pltk = parent.pltk
                    self.pos = parent.pos
                elif isinstance(parent, FSMUtil.PeekableLazyTokenizer):
                    self._init(parent._it)

            def next(self):
                ret = self.pltk.getByTokenAbsIndex(self.pos)
                self.movNext()
                return ret

            def movNext(self):
                self.pos += 1

            def movPrev(self):
                self.pos -= 1

        Peeker = Iterator

        s: str
        matchers: list["FSMUtil.TokenMatcher"]
        _tokenList: list["FSMUtil.Token"]
        _indexTextTokenizing: int
        _it: Iterator

        def __init__(
            self,
            s: str,
            matchers: list["FSMUtil.TokenMatcher"],
            start=0,
        ):
            self.s = s
            self.matchers = matchers
            self._tokenList = list()
            self._indexTextTokenizing = start
            self._it = self.Iterator(None)
            self._it.pltk = self
            self._it.pos = 0

        def _tokenizeNext(self):
            token = FSMUtil.getToken(self.s, self._indexTextTokenizing, self.matchers)
            self._indexTextTokenizing = token.end
            self._tokenList.append(token)

        def getByTokenAbsIndex(self, index):
            while True:
                if index < len(self._tokenList):
                    return self._tokenList[index]
                self._tokenizeNext()

        def next(self):
            return self._it.next()

        def movPrev(self):
            return self._it.movPrev()


class UrlFullResolution(LazyLoading):

    class _Scopes:
        protocol = "protocol"
        host = "host"
        path = "path"
        param = "param"
        secondaryHost = "secondaryHost"
        baseHost = "baseHost"
        domain = "domain"
        port = "port"
        folder = "folder"
        fileName = "fileName"
        fileBaseName = "fileBaseName"
        extName = "extName"

    url: str | None
    protocol = LazyLoading.LazyField(
        lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.protocol)
    )
    host = LazyLoading.LazyField(
        lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.host)
    )
    path = LazyLoading.LazyField(
        lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.path)
    )
    param = LazyLoading.LazyField(
        lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.param)
    )
    secondaryHost = LazyLoading.LazyField(
        lambda self: self._parseStepHost(UrlFullResolution._Scopes.secondaryHost)
    )
    baseHost = LazyLoading.LazyField(
        lambda self: self._parseStepHost(UrlFullResolution._Scopes.baseHost)
    )
    domain = LazyLoading.LazyField(
        lambda self: self._parseStepHost(UrlFullResolution._Scopes.domain)
    )
    port = LazyLoading.LazyField(
        lambda self: self._parseStepHost(UrlFullResolution._Scopes.port)
    )
    folder = LazyLoading.LazyField(
        lambda self: self._parseStepPath(UrlFullResolution._Scopes.folder)
    )
    fileName = LazyLoading.LazyField(
        lambda self: self._parseStepPath(UrlFullResolution._Scopes.fileName)
    )
    fileBaseName = LazyLoading.LazyField(
        lambda self: self._parseStepPath(UrlFullResolution._Scopes.fileBaseName)
    )
    extName = LazyLoading.LazyField(
        lambda self: self._parseStepPath(UrlFullResolution._Scopes.extName)
    )

    def _SetScope(self, name, val):
        self._raw_set(name, val)

    class RegPool:
        globally = regex.compile(
            r"^((?<protcol>[A-Za-z]+)://)?(?<host>[^/]+\.[^/.]+)?(?<path>[^?]*)?(\?(?<param>.*))?$"
        )
        host = regex.compile(r"^(?<host>[^:]+)(:(?<port>\d+))?$")
        path = regex.compile(
            r"^(?<folder>/?(?:[^/]+/)+)(?:(?<fileName>(?<fileBaseName>.+?)(?:\.(?<extName>.*))?))?$"
        )

    class UnexpectedException(Exception): ...

    @staticmethod
    def _parse_and_return_specified_field(func: typing.Callable):
        def f2(self: "UrlFullResolution", ret_field=None):
            func(self)
            if ret_field is not None:
                return self._raw_get(ret_field)

        return f2

    @_parse_and_return_specified_field
    def _parseStepGlobally(self):
        if any(
            (
                self._is_uninitialized(n)
                for n in [
                    UrlFullResolution._Scopes.protocol,
                    UrlFullResolution._Scopes.host,
                    UrlFullResolution._Scopes.path,
                    UrlFullResolution._Scopes.param,
                ]
            )
        ):
            protocol, host, path, param = [None] * 4
            if self.url is not None:
                matchGlobally = UrlFullResolution.RegPool.globally.match(self.url)
                if matchGlobally is not None:
                    protocol, host, path, param = matchGlobally.group(
                        "protcol", "host", "path", "param"
                    )
            self._SetScope(UrlFullResolution._Scopes.protocol, protocol)
            self._SetScope(UrlFullResolution._Scopes.host, host)
            self._SetScope(UrlFullResolution._Scopes.path, path)
            self._SetScope(UrlFullResolution._Scopes.param, param)

    @_parse_and_return_specified_field
    def _parseStepHost(self):
        self._parseStepGlobally()
        if any(
            (
                self._is_uninitialized(n)
                for n in [
                    UrlFullResolution._Scopes.port,
                    UrlFullResolution._Scopes.secondaryHost,
                    UrlFullResolution._Scopes.baseHost,
                    UrlFullResolution._Scopes.domain,
                ]
            )
        ):
            secondaryHost, baseHost, domain, port = [None] * 4
            if self.host is not None:
                matchHost = UrlFullResolution.RegPool.host.match(self.host)
                if matchHost is not None:
                    hostNoPort, port = matchHost.group("host", "port")
                    lHost = hostNoPort.split(".")
                    if len(lHost) < 2:
                        raise UrlFullResolution.UnexpectedException()
                    if not (
                        len(lHost) == 4
                        and all(str.isdigit(i) and 255 >= int(i) >= 0 for i in lHost)
                    ):
                        secondaryHost = ".".join(lHost[0:-2])
                        baseHost = ".".join(lHost[-2:])
                        domain = lHost[-1]
            self._SetScope(UrlFullResolution._Scopes.port, port)
            self._SetScope(UrlFullResolution._Scopes.secondaryHost, secondaryHost)
            self._SetScope(UrlFullResolution._Scopes.baseHost, baseHost)
            self._SetScope(UrlFullResolution._Scopes.domain, domain)

    @_parse_and_return_specified_field
    def _parseStepPath(self):
        self._parseStepGlobally()
        if any(
            (
                self._is_uninitialized(n)
                for n in [
                    UrlFullResolution._Scopes.folder,
                    UrlFullResolution._Scopes.fileName,
                    UrlFullResolution._Scopes.extName,
                ]
            )
        ):
            folder, fileName, fileBaseName, extName = [None] * 4
            if self.path is not None:
                matchPath = UrlFullResolution.RegPool.path.match(self.path)
                if matchPath is not None:
                    folder, fileName, fileBaseName, extName = matchPath.group(
                        "folder", "fileName", "fileBaseName", "extName"
                    )
            self._SetScope(UrlFullResolution._Scopes.folder, folder)
            self._SetScope(UrlFullResolution._Scopes.fileName, fileName)
            self._SetScope(UrlFullResolution._Scopes.fileBaseName, fileBaseName)
            self._SetScope(UrlFullResolution._Scopes.extName, extName)

    def calcAll(self):
        self._parseStepGlobally()
        self._parseStepHost()
        self._parseStepPath()

    def __init__(self, url: str):
        self.url = PathNormalize(url)

    @staticmethod
    def of(url: str):
        return UrlFullResolution(url)

    @staticmethod
    def of_file(filePath: str):
        return UrlFullResolution(os.path.join(f"file://fake_host/", filePath))
