import numpy as np
import typing
import dataclasses
import torch
import platform
import os
import random
import torchvision
import re
import collections
from .util_solid import perf_statistic, GetTimeString, Stream, EnsureFileDirExists


def getTorchDevice():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def getCalcDtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


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


def setModule(model: torch.nn.Module, path: str, device=None, strict=True):
    import os

    if device is None:
        device = "cpu"
    map_location = "cpu" if device == "cpu" else None
    loadmodel(model, path, map_location=map_location, strict=strict)
    return model.to(device)


type StatefulPytorchObject = torch.nn.Module | torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler


def loadmodel(
    model: StatefulPytorchObject,
    path,
    map_location=None,
    *a,
    **kw,
):
    if not os.path.exists(path):
        print(f"Warning: Loading pytorch checkpoint from path {path} not exist.")
        return
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, *a, **kw)
    return model


def savemodel(model: StatefulPytorchObject, path):
    EnsureFileDirExists(path)
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def load_state_dict_ignore_tensor_unmatched(
    model: torch.nn.Module, state_dict: dict, verbose=True
):
    """
    加载部分匹配的 state_dict，跳过形状不匹配的层
        形状校验逻辑可能不适配优化器

    Args:
        model: 目标模型
        state_dict: 预训练权重字典
        verbose: 是否打印详细信息

    Returns:
        loaded_keys: 成功加载的键列表
        skipped_keys: 跳过的键列表（形状不匹配）
        missing_keys: 模型中缺少权重的键列表
    """
    model_dict = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    missing_keys = []

    # 过滤 state_dict
    filtered_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if k in model_dict:
            model_v = model_dict[k]
            if (
                isinstance(v, torch.Tensor)
                and isinstance(model_v, torch.Tensor)
                and v.shape != model_v.shape
            ):
                skipped_keys.append(k)
                if verbose:
                    print(f"⚠️  跳过形状不匹配: {k}")
                    print(f"   检查点形状: {v.shape}")
                    print(f"   当前模型形状: {model_v.shape}")
            else:
                filtered_dict[k] = v
                loaded_keys.append(k)
        else:
            missing_keys.append(k)
            if verbose:
                print(f"ℹ️  模型中不存在该键: {k}")

    # 检查模型中有哪些键没有被加载
    unloaded_keys = set(model_dict.keys()) - set(filtered_dict.keys())
    if unloaded_keys and verbose:
        print(f"\n📋 未加载的模型参数 ({len(unloaded_keys)} 个):")
        for key in list(unloaded_keys)[:10]:  # 只显示前10个
            print(f"   - {key}")
        if len(unloaded_keys) > 10:
            print(f"   ... 还有 {len(unloaded_keys) - 10} 个")

    if verbose:
        print(f"\n✅ 可加载: {len(filtered_dict)} 个参数")
        print(f"⏭️  跳过（形状不匹配）: {len(skipped_keys)} 个参数")
        print(f"❓ 预训练中有但模型无: {len(missing_keys)} 个参数")

    # 加载过滤后的 state_dict
    model.load_state_dict(filtered_dict, strict=False)

    return filtered_dict, skipped_keys, missing_keys


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
        outfeat11: int,
        outfeatpool: int,
        outfeat33: int,
        outfeat55: int,
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
            self.path11 = (
                torch.nn.Sequential(
                    torch.nn.Conv2d(infeat, outfeat11, 1, padding="same"),
                    torch.nn.LeakyReLU(),
                )
                if outfeat11
                else None
            )
            self.pathpool = (
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(3, stride=1, padding=1),
                    torch.nn.Conv2d(infeat, outfeatpool, 1, padding="same"),
                    torch.nn.LeakyReLU(),
                )
                if outfeatpool
                else None
            )
            self.path33 = (
                torch.nn.Sequential(
                    torch.nn.Conv2d(infeat, infeat, 1, padding="same"),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(infeat, outfeat33, 3, padding="same"),
                    torch.nn.LeakyReLU(),
                )
                if outfeat33
                else None
            )
            self.path55 = (
                torch.nn.Sequential(
                    torch.nn.Conv2d(infeat, infeat, 1, padding="same"),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(infeat, outfeat55, 3, padding="same"),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(outfeat55, outfeat55, 3, padding="same"),
                    torch.nn.LeakyReLU(),
                )
                if outfeat55
                else None
            )
        elif version == "v3":
            self.path11 = (
                torch.nn.Sequential(
                    torch.nn.Conv2d(infeat, outfeat11, 1, padding="same"),
                    torch.nn.LeakyReLU(),
                )
                if outfeat11
                else None
            )
            self.pathpool = (
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(3, stride=1, padding=1),
                    torch.nn.Conv2d(infeat, outfeatpool, 1, padding="same"),
                    torch.nn.LeakyReLU(),
                )
                if outfeatpool
                else None
            )
            self.path33 = (
                torch.nn.Sequential(
                    torch.nn.Conv2d(infeat, outfeat55, 1, padding="same"),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(outfeat55, outfeat33, [1, 3], padding="same"),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv2d(outfeat33, outfeat33, [3, 1], padding="same"),
                    torch.nn.LeakyReLU(),
                )
                if outfeat33
                else None
            )
            self.path55 = (
                torch.nn.Sequential(
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
                if outfeat55
                else None
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
    def even(infeat: int, outfeat: int, bn=None, version=None):
        outfeatby4 = outfeat // 4
        leftovers = outfeat - outfeatby4 * 4
        return inception(
            infeat=infeat,
            outfeat11=outfeatby4,
            outfeatpool=outfeatby4,
            outfeat33=outfeatby4 + leftovers,
            outfeat55=outfeatby4,
            isbn=bn,
            version=version,
        )

    def forward(self, m):
        lpath_result = []
        if self.path11:
            lpath_result.append(self.path11(m))
        if self.pathpool:
            lpath_result.append(self.pathpool(m))
        if self.path33:
            lpath_result.append(self.path33(m))
        if self.path55:
            lpath_result.append(self.path55(m))
        o = torch.concat(lpath_result, dim=-3)
        if self.bn is not None:
            o = self.bn(o)
        return o  # channel


class res_through(torch.nn.Sequential):
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
        super().__init__(*components)
        if combiner is None:
            combiner = res_through.Combiner.add
        self.combiner = combiner

    def forward(self, m):
        o = m
        for i, l in enumerate(self):
            ret = l(o)
            o = self.combiner(o, ret)
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
        epoch_start=0,
        outputperbatchnum=100,
    ):
        ps = perf_statistic()
        for ep in range(epoch_start, epochnum):
            print(f"Epoch {ep} / {epochnum}")
            print("-------------------------------")
            # train
            for batch, datatuple in enumerate(dataloader):
                ps.start()
                loss = self.optimize(optimizer, datatuple)
                ps.stop().countcycle()
                if batch % outputperbatchnum == 0:
                    self.report_train_progress(ps, batch, len(dataloader), loss.item())
            self.on_epoch_finish(ep, epochnum)

        # win32api.Beep(1000, 1000)
        print("Done!")

    def report_train_progress(self, ps, batch, batch_per_epoch, loss):
        print(f"Time: {GetTimeString()}")
        print(f"Batch {batch} / {batch_per_epoch}")
        print(f"Training speed: {ps.aveTime():>5f} s/batch")
        ps.clear()
        fltloss = loss
        print(f"Instant loss: {fltloss:>7f}")
        return fltloss

    def optimize(self, optimizer, datatuple):
        loss = self.calcloss(datatuple)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def prepare(self): ...

    def calcloss(self, datatuple): ...

    def inferenceProgress(self, datatuple): ...

    def demo(self, *arg, **kw): ...
    def on_epoch_finish(self, epochnum_current, epochnum_total): ...


class ConvNormInsp(torch.nn.Module):
    def __init__(
        self,
        **kw,
    ):
        super().__init__()
        self.conv = self.get_conv(**kw)
        self.norm = self.get_norm(**kw)
        self.insp = self.get_insp(**kw)

    def get_norm(self, **kw): ...
    def get_insp(self, **kw): ...
    def get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        dtype=None,
        **kw,
    ):
        return torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dtype=dtype,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.insp is not None:
            x = self.insp(x)
        return x


class ConvGnRelu(ConvNormInsp):
    def get_norm(self, out_channels, numGroup=None, **kw):
        numGroup = numGroup or (
            4 if out_channels >= 16 else 2 if out_channels >= 8 else 1
        )
        return torch.nn.GroupNorm(numGroup, out_channels)

    def get_insp(self, **kw):
        return torch.nn.LeakyReLU()


class ConvLnRelu(ConvNormInsp):
    def get_norm(self, out_channels, normalized_shape, **kw):
        return torch.nn.LayerNorm(normalized_shape, out_channels)

    def get_insp(self, **kw):
        return torch.nn.LeakyReLU()


class ConvGnHs(ConvGnRelu):
    def get_insp(self, **kw):
        return torch.nn.Hardswish()


class ConvBnHs(ConvNormInsp):
    def get_norm(self, out_channels, **kw):
        return torch.nn.BatchNorm2d(out_channels)

    def get_insp(self, **kw):
        return torch.nn.Hardswish()


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
        return torch.mean(x, dim=(-2, -1))

    def forward(self, x):
        return GlobalAvgPooling.static_forward(x)


def setModuleFree(
    backbone: torch.nn.Module,
    freeLayers: typing.Iterable[re.Pattern] = None,
):
    freeLayers = freeLayers or []
    for name, param in backbone.named_parameters():
        if any(fl.match(name) for fl in freeLayers):
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


class PerceptualLoss:
    def get_allowed_export_pos(self) -> set: ...
    def get_model(self) -> torch.nn.Module: ...
    def exported_forward(self_percloss, self: torch.nn.Module, x: torch.Tensor): ...
    def __init__(
        self,
        export_pos: set = None,
        device: typing.Optional[str] = None,
        use_existed_weight: typing.Optional[str] = None,
    ):
        self.device = device
        self.use_existed_weight = use_existed_weight
        self.model = self.get_model().requires_grad_(False).eval().to(self.device)
        self.export_pos = export_pos or set()
        self.check_export_pos_available()

    def check_export_pos_available(self):
        allowed_export_pos = self.get_allowed_export_pos()
        assert Stream(self.export_pos).all_match(lambda x: x in allowed_export_pos)

    def __call__(self, x: torch.Tensor, xpred: torch.Tensor):
        model = self.model
        # 提取特征
        with torch.no_grad():
            lfeat_x = self.exported_forward(model, x)
        lfeat_xpred = self.exported_forward(model, xpred)
        loss_perc = (
            Stream(zip(lfeat_x, lfeat_xpred))
            .map(
                lambda feat_x, feat_xpred: torch.nn.functional.mse_loss(
                    feat_x, feat_xpred
                )
            )
            .collect(sum)
        )

        # 返回加权总损失
        return loss_perc

    def foward_gray(self, x: torch.Tensor, xpred: torch.Tensor):
        return self(x.repeat(1, 3, 1, 1), xpred.repeat(1, 3, 1, 1))

    def exportably_forward_sequential(self, seq: torch.nn.Sequential, x: torch.Tensor):
        exports: list[torch.Tensor] = []
        max_depth = max(self.export_pos)
        for ifeat, module in enumerate(seq):
            if ifeat > max_depth:
                break
            x = module(x)
            ifeat in self.export_pos and exports.append(x)
        return exports

    def _view_model_structure(self, file=None):
        print(ModuleArgDistribution(self.model, OnlyWithGrad=False), file=file)


class PerceptualLossResnet(PerceptualLoss):
    def get_allowed_export_pos(self):
        return set(range(6))

    def get_model(self):
        if self.use_existed_weight:
            model = torchvision.models.resnet18(weights=None)
            model.load_state_dict(
                torch.load(self.use_existed_weight, map_location=self.device)
            )
        else:
            model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )

        return model

    def exported_forward(
        self_percloss, self: torchvision.models.ResNet, x: torch.Tensor
    ):
        """使用resnet的前几层提取特征进行损失计算"""
        # See torchvision\models\resnet.py:ResNet._forward_impl
        max_depth = max(self_percloss.export_layers)
        exports = []
        while True:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            0 in self_percloss.export_layers and exports.append(x)
            if len(exports) >= max_depth:
                break

            x = self.layer1(x)
            1 in self_percloss.export_layers and exports.append(x)
            if len(exports) >= max_depth:
                break
            x = self.layer2(x)
            2 in self_percloss.export_layers and exports.append(x)
            if len(exports) >= max_depth:
                break
            x = self.layer3(x)
            3 in self_percloss.export_layers and exports.append(x)
            if len(exports) >= max_depth:
                break
            x = self.layer4(x)
            4 in self_percloss.export_layers and exports.append(x)
            if len(exports) >= max_depth:
                break

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            5 in self_percloss.export_layers and exports.append(x)
            if len(exports) >= max_depth:
                break

            break

        return exports


class SquezzeAndExcitation(torch.nn.Module):
    def __init__(self, in_feature: int, inner_dim: int):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_feature, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, in_feature),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = GlobalAvgPooling.static_forward(x)
        k = self.fc(pooled)
        x = x * k[:, :, None, None]
        return x


def tensor_contains_nan(x: torch.Tensor):
    return torch.any(torch.isnan(x))


def ImgTensor2NdarrayShowable(x: torch.Tensor):
    return x.cpu().numpy().transpose(0, 2, 3, 1).squeeze()


class PositionalEmbeddingSinusoidal(torch.nn.Module):
    def __init__(self, dim: int, base: int = 10000, maxlen: int = 512, device=None):
        super().__init__()
        assert dim % 2 == 0
        d = torch.arange(0, dim, step=2, device=device)[None, :]
        n = torch.arange(0, maxlen, device=device)[:, None]
        pe = torch.zeros(maxlen, dim, device=device)
        pe[:, 0::2] = torch.sin(n / base ** (d / dim))
        pe[:, 1::2] = torch.cos(n / base ** (d / dim))
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor
        self.dim = dim
        self.base = base
        self.maxlen = maxlen

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        assert N <= self.maxlen and C == self.dim
        return x + self.pe[None, :N, :]


class PositionalEmbedding2DSinusoidal(torch.nn.Module):
    def __init__(self, dim: int, base: int = 1000, maxlen: int = 32, device=None):
        super().__init__()
        assert dim % 4 == 0
        d = torch.arange(0, dim, step=4, device=device)[:, None]
        n = torch.arange(0, maxlen, device=device)[None, :]
        pe = torch.zeros(dim, maxlen, maxlen, device=device)
        pe[0::4, :, :] = torch.sin(n / base ** (d / dim))[:, :, None]
        pe[1::4, :, :] = torch.cos(n / base ** (d / dim))[:, :, None]
        pe[2::4, :, :] = torch.sin(n / base ** (d / dim))[:, None, :]
        pe[3::4, :, :] = torch.cos(n / base ** (d / dim))[:, None, :]
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor
        self.dim = dim
        self.base = base
        self.maxlen = maxlen

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H <= self.maxlen and W <= self.maxlen and C == self.dim
        return x + self.pe[None, :, :H, :W]


class MnTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        ffn_dim: int,
        num_head_q: int,
        num_head_kv: int = None,
        dropout=0.1,
        prelayer_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.atte_dim = in_dim
        assert self.atte_dim % num_head_q == 0
        self.out_dim = in_dim
        self.ffn_dim = ffn_dim
        self.num_head_q = num_head_q
        if num_head_kv is None:
            num_head_kv = num_head_q
        self.num_head_kv = num_head_kv
        assert num_head_q % num_head_kv == 0
        self.atte_dim_per_head = self.atte_dim // num_head_q
        self.v_dim_per_head = self.atte_dim_per_head
        self.dropout = dropout
        self.enable_gqa = num_head_q != num_head_kv
        self.prelayer_norm = prelayer_norm
        self.q = torch.nn.Linear(
            in_dim, self.atte_dim_per_head * self.num_head_q, bias=bias
        )
        self.k = torch.nn.Linear(
            in_dim, self.atte_dim_per_head * self.num_head_kv, bias=bias
        )
        self.v = torch.nn.Linear(
            in_dim, self.v_dim_per_head * self.num_head_kv, bias=bias
        )
        self.o = torch.nn.Linear(self.v_dim_per_head * self.num_head_q, in_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(in_dim, ffn_dim),
            self.get_actfunc(),
            torch.nn.Linear(ffn_dim, self.out_dim),
        )
        self.norm_ffn = self.get_norm(in_dim)
        self.norm_atte = self.get_norm(in_dim)

    def get_actfunc(self):
        return torch.nn.LeakyReLU()

    def get_norm(self, in_dim):
        return torch.nn.LayerNorm(in_dim)

    def selfattention(self, x: torch.Tensor, attn_mask=None, is_causal=False):
        """
        input and output shape of torch.nn.functional.scaled_dot_product_attention
        q: (N,...,Hq,L,E)
        k: (N,...,H,S,E)
        v: (N,...,H,S,Ev)
        N: Batch size...:Any number of other batch dimensions (optional)
        S: Source sequence length
        L: Target sequence length
        E: Embedding dimension of the query and key
        Ev: Embedding dimension of the value
        Hq: Number of heads of query
        H: Number of heads of key and value
        """
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_head_q, self.atte_dim_per_head)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_head_kv, self.atte_dim_per_head)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_head_kv, self.v_dim_per_head)
            .permute(0, 2, 1, 3)
        )
        r = (
            torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout,
                enable_gqa=self.enable_gqa,
                is_causal=is_causal,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, N, self.v_dim_per_head * self.num_head_q)
        )
        o = self.o(r)
        return o

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=False):
        if self.prelayer_norm:
            # prelayer norm
            x = (
                self.selfattention(
                    self.norm_atte(x), attn_mask=attn_mask, is_causal=is_causal
                )
                + x
            )
            x = self.ffn(self.norm_ffn(x)) + x
        else:
            x = self.norm_atte(
                self.selfattention(x, attn_mask=attn_mask, is_causal=is_causal) + x
            )
            x = self.norm_ffn(self.ffn(x) + x)
        return x


class RoPE(torch.nn.Module):
    def __init__(self, dim: int, maxlen=64, base=10000):
        super().__init__()
        self.dim = dim
        self.maxlen = maxlen
        self.base = base
        cos, sin = self._getEmbCoeff()
        self.register_buffer("cos", cos, persistent=False)
        self.cos: torch.Tensor
        self.register_buffer("sin", sin, persistent=False)
        self.sin: torch.Tensor

    def _getEmbCoeff(self):
        m = torch.arange(self.maxlen)[None, :, None]
        i = torch.arange(0, self.dim, step=2)[None, None, :]
        theta = self.base ** (-i / self.dim)
        return torch.cos(m * theta), torch.sin(m * theta)

    def forward(self, x: torch.Tensor):
        B, L, E = x.shape
        r = torch.zeros_like(x)
        cos = self.cos[:, :L, :]
        sin = self.sin[:, :L, :]
        r[:, :, 0::2] = x[:, :, 0::2] * cos - x[:, :, 1::2] * sin
        r[:, :, 1::2] = x[:, :, 1::2] * cos + x[:, :, 0::2] * sin
        return r


class RoPE2D(RoPE):
    def _getEmbCoeff(self):
        m = torch.arange(self.maxlen)[None, :, None]
        i = torch.arange(0, self.dim, step=4)[None, None, :]
        theta = self.base ** (-i / self.dim)
        return torch.cos(m * theta), torch.sin(m * theta)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        H, W, E = shape[-3:]
        cos = self.cos
        sin = self.sin
        w = torch.arange(0, W)
        h = torch.arange(0, H)
        x = x.reshape(-1, H, W, E)
        r = torch.zeros_like(x)
        r[:, :, :, 0::4] = (
            x[:, :, :, 0::4] * cos[:, :H, None, :]
            - x[:, :, :, 1::4] * sin[:, :H, None, :]
        )
        r[:, :, :, 1::4] = (
            x[:, :, :, 1::4] * cos[:, :H, None, :]
            + x[:, :, :, 0::4] * sin[:, :H, None, :]
        )
        r[:, :, :, 2::4] = (
            x[:, :, :, 2::4] * cos[:, None, :W, :]
            - x[:, :, :, 3::4] * sin[:, None, :W, :]
        )
        r[:, :, :, 3::4] = (
            x[:, :, :, 3::4] * cos[:, None, :W, :]
            + x[:, :, :, 2::4] * sin[:, None, :W, :]
        )
        r = r.reshape(*shape)
        return r
