import dataclasses
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def NewPyPlotAxis():
    fig, ax = plt.subplots()
    return ax


class nestedPyPlot:
    def __init__(self, outtershape, innershape, fig) -> None:
        self.oshape = np.array(outtershape)
        self.ishape = np.array(innershape)
        self.realsize = self.oshape * self.ishape
        self.fig = fig

    def subplot(self, o, i):
        maincoor = np.array((int(o / self.oshape[1]), o % self.oshape[1]))
        subcoor = np.array((int(i / self.ishape[1]), i % self.ishape[1]))
        realcoor = self.ishape * maincoor + subcoor
        # plt.subplot(self.realsize[0], self.realsize[1], realcoor[0]
        #             * self.realsize[1]+realcoor[1]+1)
        ax = self.fig.add_subplot(
            self.realsize[0],
            self.realsize[1],
            realcoor[0] * self.realsize[1] + realcoor[1] + 1,
        )
        return ax


class MassivePicturePlot:
    @staticmethod
    def SquarePlotShape(plotNum):
        return [int(np.ceil(np.sqrt(plotNum)))] * 2

    def __init__(self, plotShape, fig=None):
        self.plotShape = plotShape
        self.fig = fig if fig else plt.figure(figsize=5 * np.array(plotShape))
        self.i = 1

    def toNextPlot(self) -> plt.Axes:
        if self.isFull():
            raise IndexError("Too many pictures")
        ax = self.fig.add_subplot(self.plotShape[0], self.plotShape[1], self.i)
        self.i += 1
        return ax

    def isFull(self):
        return self.i > np.prod(self.plotShape)

    def show(self, img_ndarray):
        self.toNextPlot().imshow(img_ndarray)


@dataclasses.dataclass
class MassivePicturePlotSubimged(MassivePicturePlot):
    mainGrid: list
    subimgPerInstance: int
    fig: plt.Figure = None

    def __post_init__(self):
        y, x = self.mainGrid
        super().__init__([y, self.subimgPerInstance * x], fig=self.fig)


FloatImgPltImshowConfig = {"vmin": 0, "vmax": 1}
