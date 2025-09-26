import numpy as np
import typing
import dataclasses
import torch
import platform
import os
import random
from .util_solid import perf_statistic


def getTorchDevice():
    print(getDeviceInfo())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device


def getDeviceInfo():
    try:
        import psutil

        # 获取CPU信息
        cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
        threads_count = psutil.cpu_count(logical=True)  # 逻辑线程数
        cpu_freq = psutil.cpu_freq()  # CPU频率信息
        architecture = platform.architecture()[0]
        processor = platform.processor()
        mem = psutil.virtual_memory()
        cpuInfo = (
            f"CPU总内存: {mem.total/1024**3} GB\n\n"
            + f"CPU可用内存: {mem.available/1024**3} GB\n"
            + f"CPU使用率: {mem.percent}%\n"
            + f"CPU物理核心数: {cpu_count}\n"
            + f"CPU逻辑线程数: {threads_count}\n"
            + f"CPU 架构: {architecture}\n"
            + f"CPU 型号: {processor}\n"
            + f"CPU频率: 当前 {cpu_freq.current:.2f} MHz, 最小 {cpu_freq.min:.2f} MHz, 最大 {cpu_freq.max:.2f} MHz\n"
        )
    except ImportError:
        # Get CPU info
        cpu_info = platform.processor()
        num_cores = os.cpu_count()
        architecture = platform.architecture()[0]
        processor = platform.processor()
        cpu_frequency = "Not available in Python standard library"
        cpuInfo = (
            f"CPU Info: {cpu_info}, Cores: {num_cores}, Frequency: {cpu_frequency}\n"
        )

    try:
        gpu_name = torch.cuda.get_device_name()
    except:
        gpu_name = "No GPU detected"
    gpuInfo = f"GPU Name: {gpu_name}\n"

    return f"{cpuInfo}\n" + f"{gpuInfo}"


def spectrumDecompose(s, psize):
    if type(s) is int:
        s = torch.tensor([s])
    batchsize = s.shape[0]
    s = s.reshape((batchsize, 1))
    return torch.zeros([batchsize, psize], dtype=torch.float).scatter_(
        dim=-1, index=s, src=torch.ones_like(s, dtype=torch.float)
    )


def batchsizeof(tensor):
    return tensor.shape[0]


def setModule(model: torch.nn.Module, path=None, device=None, strict=True):
    import os

    if device is None:
        device = "cpu"

    if path is None:
        print(f"Path==None")
    elif not os.path.exists(path):
        print(f"Warning: Path {path} not exist. Set model default")
    else:
        print(f"Loading existed nn {path}")
        model.load_state_dict(
            torch.load(path, map_location=torch.device(device), weights_only=True),
            strict=strict,
        )
    return model.to(device)


def savemodel(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def tensorimg2ndarray(m: torch.Tensor):
    m = m.cpu().numpy()
    if not len(m.shape) == 2:  # not single channeled
        m = np.moveaxis(m, -3, -1)
    return m


class skiper(torch.nn.Module):
    def __init__(self, component, n_i, n_o) -> None:
        super().__init__()
        self.component = component
        self.combiner = torch.nn.Conv2d(n_i + n_o, n_o, 1)

    def forward(self, m):
        # [b,c,h,w]
        processed = self.component.forward(m)
        c = torch.concat([processed, m], dim=-3)
        result = self.combiner.forward(c)
        return result


class cbr(torch.nn.Module):
    def __init__(self, n_i, n_o, n_c) -> None:
        super().__init__()
        self.component = torch.nn.Sequential(
            torch.nn.Conv2d(n_i, n_o, n_c, padding="same", bias=False),
            torch.nn.BatchNorm2d(n_o),
            torch.nn.LeakyReLU(),
        )

    def forward(self, m):
        # [b,c,h,w]
        return self.component.forward(m)


class cbrps(torch.nn.Module):
    # input chan, output chan, convolve size, pooling size
    # n_o should be like 2*n, cuz maxpool will be concated with former output
    def __init__(self, n_i, n_o, n_c, n_p) -> None:
        super().__init__()
        self.component = torch.nn.Sequential(
            torch.nn.Conv2d(n_i, n_o, n_c, padding="same", bias=False),
            torch.nn.BatchNorm2d(n_o),
            torch.nn.LeakyReLU(),
            skiper(torch.nn.MaxPool2d(n_p, stride=1, padding=int(n_p / 2)), n_o, n_o),
        )

    def forward(self, m):
        # [b,c,h,w]
        return self.component.forward(m)


class inception(torch.nn.Module):
    def __init__(
        self,
        infeat,
        outfeat11,
        outfeatpool,
        outfeat33,
        outfeat55,
        isbn=True,
        version=None,
    ) -> None:
        super().__init__()
        self.infeat = infeat
        self.outfeat11 = outfeat11
        self.outfeatpool = outfeatpool
        self.outfeat33 = outfeat33
        self.outfeat55 = outfeat55
        self.isbn = isbn
        if version is None:
            version = "v2"
        self.version = version
        if version == "v2":
            self.path11 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat11, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.pathpool = torch.nn.Sequential(
                torch.nn.MaxPool2d(3, stride=1, padding=1),
                torch.nn.Conv2d(infeat, outfeatpool, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path33 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, infeat, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(infeat, outfeat33, 3, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path55 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, infeat, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(infeat, outfeat55, 3, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, 3, padding="same"),
                torch.nn.LeakyReLU(),
            )
        elif version == "v3":
            self.path11 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat11, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.pathpool = torch.nn.Sequential(
                torch.nn.MaxPool2d(3, stride=1, padding=1),
                torch.nn.Conv2d(infeat, outfeatpool, 1, padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path33 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat55, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat33, [1, 3], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat33, outfeat33, [3, 1], padding="same"),
                torch.nn.LeakyReLU(),
            )
            self.path55 = torch.nn.Sequential(
                torch.nn.Conv2d(infeat, outfeat55, 1, padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [1, 3], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [3, 1], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [1, 3], padding="same"),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(outfeat55, outfeat55, [3, 1], padding="same"),
                torch.nn.LeakyReLU(),
            )
        else:
            raise ValueError(f"{version} not supported")
        if isbn is not None and isbn:
            self.bn = torch.nn.BatchNorm2d(
                outfeat11 + outfeatpool + outfeat33 + outfeat55
            )
        else:
            self.bn = None

    @staticmethod
    def even(infeat, outfeat, bn=None, version=None):
        assert outfeat % 4 == 0
        outfeatby4 = outfeat // 4
        return inception(
            infeat, outfeatby4, outfeatby4, outfeatby4, outfeatby4, bn, version
        )

    def forward(self, m):
        o = torch.concat(
            [self.path11(m), self.pathpool(m), self.path33(m), self.path55(m)], dim=-3
        )
        if self.bn is not None:
            o = self.bn(o)
        return o  # channel


class res_through(torch.nn.Module):
    class Combiner:
        @staticmethod
        def add(last, current):
            return last + current

        class concat:
            def __init__(self, dim=1):
                self.dim = dim

            def __call__(self, last, current):
                return torch.concat([last, current], dim=self.dim)

    def __init__(self, *components, combiner: "res_through.Combiner" = None) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(*components)
        if combiner is None:
            combiner = res_through.Combiner.add
        self.combiner = combiner

    def forward(self, m):
        o = m
        for i, l in enumerate(self.seq):
            ret = l(o)
            o = self.combiner(o, ret)
        return o


class OneShotAggregationResThrough(torch.nn.Module):
    def __init__(self, *components, chanTotal, chanDest) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(*components)
        self.combiner = torch.nn.Conv2d(chanTotal, chanDest, 1)

    def forward(self, m):
        o = [m]
        t = m
        for i, l in enumerate(self.seq):
            t = l(t)
            o.append(t)
        o = self.combiner(torch.concat(o, dim=1))
        return o


class ModuleFunc(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime


import time
import typing


class trainpipe:
    def train(
        self,
        dataloader,
        optimizer,
        epochnum=10,
        outputperbatchnum=100,
        customSubOnOutput=None,
    ):
        start_time = time.time()
        ps = perf_statistic()
        for ep in range(epochnum):
            print(f"Epoch {ep} / {epochnum}")
            print("-------------------------------")

            # train
            for batch, datatuple in enumerate(dataloader):
                ps.start()
                loss = self.trainprogress(datatuple)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ps.stop().countcycle()
                if batch % outputperbatchnum == 0:
                    print(f"Batch {batch} / {len(dataloader)}")
                    print(f"Training speed: {ps.aveTime():>5f} seconds per batch")
                    ps.clear()
                    aveloss = loss.item()
                    print(f"Instant loss: {aveloss:>7f}")
                    self.train_progress_echo(batch=batch, loss=aveloss)

        # win32api.Beep(1000, 1000)
        print("Done!")

    def prepare(self): ...

    def train_progress_echo(self, batch, loss): ...

    def calcloss(self, *arg, **kw): ...

    def trainprogress(self, datatuple): ...

    def inferenceProgress(self, datatuple): ...

    def demo(self, *arg, **kw): ...


class ConvNormInsp(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        norm=None,
        insp=None,
        dtype=None,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dtype=dtype,
        )
        self.norm = norm
        self.insp = insp

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.insp(x)
        return x


class ConvGnRelu(ConvNormInsp):
    def __init__(self, in_channels, out_channels, numGroup=4, *a, **kw):
        super().__init__(
            in_channels,
            out_channels,
            norm=torch.nn.GroupNorm(numGroup, out_channels),
            insp=torch.nn.LeakyReLU(),
            *a,
            **kw,
        )


class ConvGnHs(ConvNormInsp):
    def __init__(self, in_channels, out_channels, numGroup=4, *a, **kw):
        super().__init__(
            in_channels,
            out_channels,
            norm=torch.nn.GroupNorm(numGroup, out_channels),
            insp=torch.nn.Hardswish(),
            *a,
            **kw,
        )


class ConvBnHs(ConvNormInsp):
    def __init__(self, in_channels, out_channels, *a, **kw):
        super().__init__(
            in_channels,
            out_channels,
            norm=torch.nn.BatchNorm2d(out_channels),
            insp=torch.nn.Hardswish(),
            *a,
            **kw,
        )


class OneShotAggregationResThrough(torch.nn.Module):
    def __init__(self, *components, chanTotal, chanDest) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(*components)
        self.combiner = torch.nn.Conv2d(chanTotal, chanDest, 1)

    def forward(self, m):
        o = [m]
        t = m
        for i, l in enumerate(self.seq):
            t = l(t)
            o.append(t)
        o = self.combiner(torch.concat(o, dim=1))
        return o


class ModuleFunc(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def ModuleArgDistribution(mod: torch.nn.Module, OnlyWithGrad: bool = True):
    cond = (lambda v: v.requires_grad) if OnlyWithGrad else (lambda v: True)
    return "\n".join(
        [f"{k}: {v.numel()}" for k, v in mod.named_parameters() if cond(v)]
    )


class GlobalAvgPooling(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def static_forward(x):
        return torch.mean(x, dim=(2, 3))

    def forward(self, x):
        return GlobalAvgPooling.static_forward(x)


def setModuleFree(backbone: torch.nn.Module, freeLayers: typing.Iterable):
    for name, param in backbone.named_parameters():
        if any([name.startswith(fl) for fl in freeLayers]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return backbone


def getmodel(model0: torch.nn.Module, *arg, **kwarg):
    model = setModule(model0, *arg, **kwarg)
    paramNum = np.sum(
        [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    )
    print(f"{paramNum=}")
    # print(model)
    return model


class FinalModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("opt_step", torch.tensor(0, dtype=torch.int32))
        self.opt_step: torch.Tensor

    def update_step(self, delta=1):
        self.opt_step += delta

    def parameters(
        self, recurse: bool = True
    ) -> typing.Iterator[torch.nn.parameter.Parameter]:
        return filter(
            lambda x: x.requires_grad is not False, super().parameters(recurse)
        )

    def load(self, path, *a, **kw):
        getmodel(self, path, *a, **kw)
        return self

    def save(self, path):
        savemodel(self, path)


class MPn(torch.nn.Module):
    def __init__(self, in_channels, n_value=1, downSamplingStride=2):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels % 2 == 0
        out_channels = n_value * in_channels
        self.out_channels = out_channels
        cPath = out_channels // 2
        self.wayPooling = torch.nn.Sequential(
            torch.nn.MaxPool2d(downSamplingStride, downSamplingStride),
            ConvGnHs(in_channels, cPath),
        )
        self.wayConv = torch.nn.Sequential(
            ConvGnHs(in_channels, cPath, kernel_size=1),
            ConvGnHs(
                cPath,
                cPath,
                stride=downSamplingStride,
                padding=1,
                kernel_size=downSamplingStride + 1,
            ),
        )
        self.combiner = ConvGnHs(cPath * 2, out_channels)

    def forward(self, x):
        o_pool = self.wayPooling(x)
        o_conv = self.wayConv(x)
        return self.combiner(torch.concat([o_pool, o_conv], dim=1))


class OnlineGeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def generate(self):
        raise NotImplementedError()

    def __getitem__(self, index: int):
        return self.generate()

    def __len__(self):
        return self.length


@dataclasses.dataclass
class Deterministic:
    seed: int

    def do(self):
        seed = self.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return self

    def worker_init_fn(self):
        """
        use in dataloader like:
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                num_workers=4,
                worker_init_fn=worker_init_fn
            )
        """

        def fn(worker_id):
            np.random.seed(self.seed + worker_id)

        return fn
