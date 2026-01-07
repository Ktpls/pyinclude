import typing
import os
import zipfile
import re


def AllFileIn(
    path,
    includeFileInSubDir=True,
    path_filter: typing.Optional[typing.Callable[[str], bool]] = None,
):
    for dirpath, dir, file in os.walk(path):
        if not includeFileInSubDir and dirpath != path:
            continue
        fullPath = map(lambda f: os.path.join(dirpath, f), file)
        if path_filter is not None:
            fullPath = filter(path_filter, fullPath)
        yield from fullPath


def ReadFile(path):
    with open(path, "rb") as f:
        return f.read()


def EnsureDirectoryExists(directory):
    if len(directory) == 0:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)


def EnsureFileDirExists(path):
    EnsureDirectoryExists(os.path.dirname(path))


def WriteFile(path, content):
    EnsureFileDirExists(path)
    with open(path, "wb+") as f:
        f.write(content)


def AppendFile(path, content):
    EnsureFileDirExists(path)
    with open(path, "ab+") as f:
        f.write(content)


def ReadTextFile(path: str, encoding="utf-8") -> str:
    return ReadFile(path).decode(encoding)


def WriteTextFile(path: str, text: str, encoding="utf-8"):
    WriteFile(path, text.encode(encoding))


def PathNormalize(path: str):
    return path.replace("\\", "/")


def ReadFileInZip(zipf, filename: str | list[str] | tuple[str]):
    zipf = zipfile.ZipFile(zipf)
    singleFile = not isinstance(filename, (tuple, list))
    if singleFile:
        filename = [filename]
    file = [zipf.read(f) for f in filename]
    if singleFile:
        return file[0]
    return file


def make_filename_safe(filename):
    # Remove any characters that are not allowed in a filename
    filename = re.sub(r'[<>:"/\\|?*\r\n]', "_", filename)

    # Replace any consecutive invalid characters with a single underscore
    # filename = re.sub(r'(.)\1+', r'\1', filename)

    return filename
