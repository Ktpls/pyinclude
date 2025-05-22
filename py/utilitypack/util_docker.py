import os
from utilitypack.util_solid import *


class DockerBuilder:
    imgName: str = None
    containerName: str = None
    port: dict[str, str] = None
    mount: dict[str, str] = None
    env: dict[str, str] = None
    network: list[str] = None
    runArg: list[str] = None
    docker_file_dir: str = None
    docker_compose_dir: str = None
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

    def _toDir(self, path):
        os.chdir(self.cwd), os.chdir(path)

    def build(self):
        os.system(f"docker build -t {self.imgName} {self.docker_file_dir}")
        return self

    @CwdProtected
    def docker_compose_build(self):
        if self.docker_compose_dir:
            self._toDir(self.docker_compose_dir)
        os.system(f"docker compose build")
        return self

    @CwdProtected
    def recompose(self):
        if self.docker_compose_dir:
            self._toDir(self.docker_compose_dir)
        os.system("docker compose down")
        os.system("docker compose up -d")
        return self

    def export(self):
        os.system(f"docker save -o {self.img_file_name} {self.imgName}")
        return self

    def stopcontainer(self):
        try:
            os.system(f"docker stop {self.containerName}")
        except:
            pass
        return self

    def rmcontainer(self):
        try:
            os.system(f"docker rm {self.containerName}")
        except:
            pass
        return self

    def run(self):
        params = [
            f"-it -d --name {self.containerName}",
        ]
        if self.port:
            params.extend([f"-p {k}:{v}" for k, v in self.port.items()])
        if self.mount:
            params.extend([f"-v {k}:{v}" for k, v in self.mount.items()])
        if self.env:
            params.extend([f"-e {k}={v}" for k, v in self.env.items()])
        if self.network:
            params.extend([f"--network {v}" for v in self.network])
        params.append(f"{self.imgName}")
        if self.runArg:
            params.extend(self.runArg)

        cmd = " ".join(params)
        cmd = f"docker run {cmd}"
        os.system(cmd)
        return self

    def restart(self):
        os.system(f"docker restart {self.containerName}")
        return self

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
            PythonFrontEnd,
            CLikeFrontEnd,
            AsmFrontEnd,
            PowerDefineBlockParser,
            PowerDefineEnviroment,
        )

        front_end_mapping = {
            "yaml": YamlFrontEnd,
            "yml": YamlFrontEnd,
            "py": PythonFrontEnd,
            "c": CLikeFrontEnd,
            "cpp": CLikeFrontEnd,
            "asm": AsmFrontEnd,
        }
        for p in files:
            if isinstance(p, (tuple, list)):
                p, extName = p
            else:
                extName = UrlFullResolution.of(p).extName
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
