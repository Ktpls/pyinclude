import os
import subprocess
from utilitypack.util_solid import (
    ReadTextFile,
    WriteTextFile,
    UrlFullResolution,
    Stream,
    AllFileIn,
)
import types
from utilitypack.cold.util_solid import DistillLibraryFromDependency


class DockerBuilder:
    imgName: str = None
    containerName: str = None
    port: dict[str, str] = None
    mount: dict[str, str] = None
    env: dict[str, str] = None
    network: list[str] = None
    runArg: list[str] = None
    dockerfile_dir: str = None
    dockerfile_path: str = None
    dockercompose_dir: str = None
    img_file_name: str = "image.tar"

    def __init__(self):
        self.cwd = os.getcwd()

    @staticmethod
    def CwdProtected(f=None):
        def toGetF(f):
            def newF(*a, **kw):
                cwd = os.getcwd()
                ret = f(*a, **kw)
                os.chdir(cwd)
                return ret

            return newF

        if f is None:
            return toGetF
        else:
            return toGetF(f)

    def to_dir(self, path):
        path and (os.chdir(self.cwd), os.chdir(path))
        return self

    def build(self, imgName=None, target=None):
        cmd = ["docker", "build"]
        if target:
            cmd.extend(["--target", target])
        if imgName := imgName or self.imgName:
            cmd.extend(["-t", imgName])
        if self.dockerfile_path:
            cmd.extend(["-f", self.dockerfile_path])
        cmd.append(self.dockerfile_dir or ".")
        subprocess.run(cmd, check=True)
        return self

    @CwdProtected
    def docker_compose_build(self):
        if self.dockercompose_dir:
            self.to_dir(self.dockercompose_dir)
        subprocess.run(["docker", "compose", "build"], check=True)
        return self

    @CwdProtected
    def recompose(self, remove_orphans: bool = None):
        return self.compose_down().compose_up()

    @CwdProtected
    def compose_down(self, remove_orphans: bool = None):
        self.dockercompose_dir and self.to_dir(self.dockercompose_dir)
        cmd = ["docker", "compose", "down"]
        remove_orphans and cmd.append("--remove-orphans")
        subprocess.run(cmd, check=True)
        return self

    @CwdProtected
    def compose_up(self, remove_orphans: bool = None):
        self.dockercompose_dir and self.to_dir(self.dockercompose_dir)
        cmd = ["docker", "compose", "up", "-d"]
        remove_orphans and cmd.append("--remove-orphans")
        subprocess.run(cmd, check=True)
        return self

    def export(self, imgName=None, img_file_name=None):
        img_file_name = img_file_name or self.img_file_name
        imgName = imgName or self.imgName
        cmd = ["docker", "save", "-o", img_file_name, imgName]
        subprocess.run(cmd, check=True)
        return self

    def stopcontainer(self):
        try:
            subprocess.run(["docker", "stop", self.containerName], check=True)
        except:
            pass
        return self

    def rmcontainer(self):
        try:
            subprocess.run(["docker", "rm", self.containerName], check=True)
        except:
            pass
        return self

    def run(self):
        params = [
            "-it",
            "-d",
            f"--name={self.containerName}",
        ]
        if self.port:
            for k, v in self.port.items():
                params.append(f"-p{k}:{v}")
        if self.mount:
            for k, v in self.mount.items():
                params.append(f"-v{k}:{v}")
        if self.env:
            for k, v in self.env.items():
                params.append(f"-e{k}={v}")
        if self.network:
            for v in self.network:
                params.extend(["--network", v])
        params.append(self.imgName)
        if self.runArg:
            params.extend(self.runArg)

        subprocess.run(["docker", "run"] + params, check=True)
        return self

    def restart(self):
        subprocess.run(["docker", "restart", self.containerName], check=True)
        return self

    NORMAL_HASH_COMMENTED_FILE = "normal_hash_commented_file"

    def preproc_pd(self, files: list[str | tuple[str, str]], pdenv: dict = None):
        """
        预处理PowerDefine模板文件

        Args:
            files: 需要处理的文件列表，支持两种格式：
                - str: 文件路径 (自动提取扩展名)
                - tuple[str, str]: (文件路径, 指定扩展名)
                支持扩展名: yaml/yml/py/c/cpp/asm
                示例: ["config.yml", ("src/main", "c"), "kernel.asm"]

            pdenv: PowerDefine环境变量字典，用于模板预处理
                示例: {"DEBUG_MODE": "1", "MAX_THREADS": "4"}

        Returns:
            self: 支持方法链式调用

        Note:
            - 会直接修改原始文件内容
            - 文件扩展名必须在支持列表中
            - 使用UrlFullResolution解析文件路径
        """
        from powerDefine import (
            YamlFrontEnd,
            NormalHashCommentFrontEnd,
            PythonFrontEnd,
            CLikeFrontEnd,
            AsmFrontEnd,
            PowerDefineBlockParser,
            PowerDefineEnviroment,
        )

        front_end_mapping = {
            "yaml": YamlFrontEnd,
            "yml": YamlFrontEnd,
            self.NORMAL_HASH_COMMENTED_FILE: NormalHashCommentFrontEnd,
            "py": PythonFrontEnd,
            "c": CLikeFrontEnd,
            "cpp": CLikeFrontEnd,
            "asm": AsmFrontEnd,
        }
        for p in files:
            if isinstance(p, (tuple, list)) and len(p) == 2:
                p, extName = p
            else:
                extName = os.path.splitext(p)[1][1:].lower()
            assert extName in front_end_mapping
            pd_cwd = UrlFullResolution.of(p).folder
            pde = PowerDefineEnviroment(env=pdenv, cwd=pd_cwd)
            pdbp = PowerDefineBlockParser(pde)
            fe = front_end_mapping[extName](pdbp)
            s = ReadTextFile(p)
            s = fe.preproc(s)
            WriteTextFile(p, s)
            print(f"File edited: {p}")
        return self

    def DistillUtils(
        self, your_project: str, util_module: types.ModuleType, target_file: str
    ):
        target_file_not_exists = not os.path.exists(target_file)
        uts_copy = DistillLibraryFromDependency.DistillLibrary(
            Stream(AllFileIn(your_project))
            .filter(lambda x: not x.startswith("_"))
            .filter(lambda x: x.endswith(".py"))
            .filter(
                lambda x: target_file_not_exists or not os.path.samefile(x, target_file)
            )
            .sorted()
            .map(lambda x: ReadTextFile(x)),
            Stream(AllFileIn(UrlFullResolution.of(util_module.__file__).folder))
            .filter(lambda x: not x.startswith("_") and x.endswith(".py"))
            .sorted()
            .map(lambda x: ReadTextFile(x)),
        )
        uts_copy = (
            """\
try:from utilitypack.util_solid import *
except:...
"""
            + uts_copy
        )
        WriteTextFile(target_file, uts_copy)
        return self


def ExportDotEnvToDockerCompose(env_file: str, indent: str = "") -> str:
    import re

    r = (
        Stream(
            re.finditer(
                r"^(?P<k>[A-Za-z0-9_]+)=(?P<v>.*?)(?: #.*)?$",
                ReadTextFile(env_file),
                re.MULTILINE,
            )
        )
        .map(lambda x: "%s: ${%s:-%s}" % (x.group("k"), x.group("k"), x.group("v")))
        .map(lambda x: f"{indent}{x}")
        .collect(lambda x: "\n".join(x))
    )
    return r
