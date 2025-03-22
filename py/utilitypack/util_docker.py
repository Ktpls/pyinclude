import os


class Builder:
    imgName: str
    containerName: str

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

    @CwdProtected
    def generateDockerCompose(self):
        self._toDir("docker")
        os.system("python generate_docker_compose")
        return self

    @CwdProtected
    def build(self, root_dir: str = None):
        if root_dir:
            self._toDir(root_dir)
        os.system(f"docker build -t {self.imgName} .")
        return self

    @CwdProtected
    def recompose(self, docker_compose_dir: str = None):
        if docker_compose_dir:
            self._toDir(docker_compose_dir)
        os.system("docker compose down")
        os.system("docker compose up -d")
        return self

    def export(self, fileName: str = None):
        fileName = fileName or "image.tar"
        os.system(f"docker save -o {fileName} {self.imgName}")
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

    def run(
        self,
        port: dict[str, str] = None,
        mount: dict[str, str] = None,
        env: dict[str, str] = None,
    ):
        params = [
            f"-it -d --name {self.containerName}",
        ]
        if port:
            params.extend([f"-p {k}:{v}" for k, v in port.items()])
        if mount:
            params.extend([f"-v {k}:{v}" for k, v in mount.items()])
        if env:
            params.extend([f"-e {k}={v}" for k, v in env.items()])
        params.append(f"{self.imgName}")

        cmd = " ".join(params)
        cmd = f"docker run {cmd}"
        os.system(cmd)
        return self

    def restart(self):
        os.system(f"docker restart {self.containerName}")
        return self
