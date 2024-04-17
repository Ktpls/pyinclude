from .util_solid import *
from .util_np import *

"""
opencv
"""

import cv2 as cv


def savemat(m, name=None, path=None, autorename=True):
    if path is None:
        path = r"./output/"
    defaultName = "unnamed"
    defaultSuffix = ".png"
    if name is None:
        name = defaultName + defaultSuffix
    namesplit = os.path.splitext(name)
    name, suffix = str(namesplit[0]), str(namesplit[1])
    if len(suffix) == 0:
        suffix = defaultSuffix

    if not os.path.exists(path):
        os.makedirs(path)
    totalpath = os.path.join(path, name + suffix)
    # find suitable name
    if autorename and os.path.exists(totalpath):
        suffix_idx = 0
        while True:
            suffix_idx += 1
            newname = "{}-{}".format(name, suffix_idx)
            totalpath = os.path.join(path, newname + suffix)
            if not os.path.exists(totalpath):
                break

    if not cv.imwrite(totalpath, m):
        raise IOError(f"Bad write {totalpath}")


def savematn(m: np.ndarray, name=None, path=None):
    mtmp = m.copy()
    cv.normalize(mtmp, mtmp, 0, 255, cv.NORM_MINMAX)
    savemat(mtmp, name, path)


def savematflt(m, multiplier=255, name=None, path=None):
    savemat(multiplier * m, name, path)


def regionsum(m, size, mask=None):
    if m.size <= 0:
        return m
    if mask is not None:
        mask[mask > 0] = 1
    if len(m.shape) > 2 and mask is not None:  # with channel dim
        mask = mask.reshape(mask.shape + (1,))
    return cv.filter2D(m if mask is None else m * mask, -1, np.ones(size, np.float32))


def regionave(m, size, mask=None, notConsiderMaskInDenominator=True):
    """
    if notConsiderMaskInDenominator:
    denominator will not consider mask and boundary and be size[0]*size[1]
    else:
    denominator will be #pix nearby on mask
    u may ask mask==None does the same as notConsiderMaskInDenominator==True
    but if u want to use mask and dont want to be constrained by boundary. unimplemented though
    """

    if m.size <= 0:
        return m
    if mask is not None:
        mask = np.copy(mask)
        mask[mask > 0] = 1
    if mask is None or notConsiderMaskInDenominator:
        denominator = size[0] * size[1]
    else:
        denominator = regionsum(mask, size) + 0.01
        if len(m.shape) > 2:  # m with channel dim
            denominator = denominator.reshape(denominator.shape + (1,))
    return regionsum(m, size, mask) / denominator


def density(p, size):
    return regionave(p.astype("float"), size)


def densityfilter(p, size, thresh):
    dence = density(p, size)
    return np.logical_and(p, dence >= thresh)


rgb2hsvmat = np.array(
    [
        [
            [np.cos(0), np.cos(2 / 3 * np.pi), np.cos(4 / 3 * np.pi)],
            [np.sin(0), np.sin(2 / 3 * np.pi), np.sin(4 / 3 * np.pi)],
            [1, 0, 0],
        ],
        [
            [np.cos(0), np.cos(2 / 3 * np.pi), np.cos(4 / 3 * np.pi)],
            [np.sin(0), np.sin(2 / 3 * np.pi), np.sin(4 / 3 * np.pi)],
            [0, 1, 0],
        ],
        [
            [np.cos(0), np.cos(2 / 3 * np.pi), np.cos(4 / 3 * np.pi)],
            [np.sin(0), np.sin(2 / 3 * np.pi), np.sin(4 / 3 * np.pi)],
            [0, 0, 1],
        ],
    ]
)
hsv2rgbmat = [np.linalg.inv(m) for m in rgb2hsvmat]


def hsv2rgb(hsv):
    h, s, v = hsv
    h = h * np.pi / 180
    xyv = np.array([s * np.cos(h), s * np.sin(h), v])

    # find the corresponding case
    for c, m in enumerate(hsv2rgbmat):
        rgb = m @ xyv
        if np.argmax(rgb) == c:
            return rgb

    # not possible, theoretically
    return np.array((0, 0, 0))

    # to view all solutions
    # rgbs=np.zeros([3,3])
    # for c,m in enumerate(mats):
    #     rgbs[c]=np.linalg.inv(m)@xyv
    # return rgbs


def rgb2hsv(rgb):
    xyv = rgb2hsvmat[np.argmax(rgb)] @ rgb
    x, y, v = xyv
    hsv = np.array([180 / np.pi * np.arctan2(y, x), np.sqrt(x**2 + y**2), v])
    return hsv


def rgb2bgr(rgb):
    m = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    return m @ rgb


def convolve_norm(m, k):
    summer = np.ones_like(k)
    mag = np.sqrt(cv.filter2D(m**2, -1, summer))
    ret = cv.filter2D(m, -1, k)
    return ret / mag


def hsv2opencv8bithsv(hsv):
    return np.array([0.5, 2.55, 2.55]) * np.array(hsv)


def outputlines2mat2(m, pos, content, textcolor=[255, 255, 255], lineinterval=10):
    # different impl., ret with content bounding box
    pos = np.array(pos).astype("int")
    line = content.split("\n")
    yoffset = 0
    xmax = 0
    fontFace = cv.FONT_HERSHEY_DUPLEX
    fontScale = 1
    thickness = 1
    for i, l in enumerate(line):
        size = np.array(cv.getTextSize(l, fontFace, fontScale, thickness)[0])
        yoffset += size[1] + lineinterval if i != 0 else size[1]
        if xmax < size[0]:
            xmax = size[0]
        m = cv.putText(
            m,
            l,
            pos + [0, yoffset],
            fontFace,
            fontScale,
            textcolor,
            thickness=thickness,
        )
    box = [pos, pos + [xmax, yoffset]]
    return m, box


def aPicWithText(
    content: str, maxsize=[1080, 1920], textcolor=[255, 255, 255], lineinterval=10
):
    """
    impl with opencv
    """
    m = np.zeros(
        maxsize
        + [
            3,
        ],
        np.uint8,
    )
    m, bbox = outputlines2mat2(m, np.array([0, 0]), content, textcolor, lineinterval)
    mshape = np.array(bbox[1]) + [0, 8]  # ret wrong for unknown reason
    m = m[: mshape[1], : mshape[0]]
    m = addShadow2HUD(m)
    return m


from PIL import Image, ImageDraw, ImageFont


def aPicWithTextWithPil(
    content: str, maxsize=[1080, 1920], textcolor=[255, 255, 255], lineinterval=10
):
    """
    impl with PIL
    """
    maxsize = list(np.flip(maxsize))
    # Create a blank image with the specified size
    image = Image.new("RGB", maxsize)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Define the font size and font type
    font_size = 30
    font = ImageFont.truetype(r"asset\common\yahei.ttf", font_size)

    # Split the content into lines
    lines = content.split("\n")

    # Draw each line of text on the image
    for iline, line in enumerate(lines):
        # Calculate the x-coordinate for the line
        x = 0
        y = iline * (font_size + lineinterval)
        # Draw the text on the image
        draw.text((x, y), line, font=font, fill=tuple(textcolor))

        # Increment the y-coordinate for the next line
        y += font_size + lineinterval

    # Return the image
    # Convert the PIL image to an np.ndarray
    return addShadow2HUD(np.array(image), 1)


def addShadow2HUD(m, thickness=2, color=50):
    gray = cv.cvtColor(m, cv.COLOR_BGR2GRAY)
    kernelshape = 2 * thickness + 1
    edgekernel = np.ones([kernelshape, kernelshape])
    edgekernel[thickness, thickness] = -100  # anchor pix must be black
    # edgekernel=np.array([
    #     [1,1,1],
    #     [1,-80,1],
    #     [1,1,1],
    # ])
    edge = cv.filter2D(gray, -1, edgekernel)
    edge = cv.threshold(edge, 0, 1, cv.THRESH_BINARY)[1]
    edge = edge.reshape(edge.shape + (1,))
    return m + edge * color


def getDemonstrationImg():
    x = np.linspace(0, 5 * 2 * np.pi, 100, dtype=np.float32).reshape(1, -1)
    y = x.T
    demo = np.sin(x + y)
    demo = ZFunc(0, 0.25, 0, 0.75)(demo) * 255
    return demo


def outputlines2mat(m, pos, content, lineheight=25, textcolor=[255, 255, 255]):
    m = m.copy()
    line = content.split("\n")
    for i, l in enumerate(line):
        cv.putText(
            m,
            l,
            pos.astype("int32") + [0, i * lineheight],
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            textcolor,
        )
    return m


class DataCollector:
    randNameLen = 10

    def __init__(self, outputpath) -> None:
        self.outputpath = outputpath

    @staticmethod
    def geneName():
        charSet4RandomString = "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return randomString(charSet4RandomString, DataCollector.randNameLen)

    @FunctionalWrapper
    def save(self, m, name=None):
        if name is None:
            name = DataCollector.geneName()
        savemat(m, f"{name}", path=self.outputpath)


def get_AABB(binary_image):
    """
    This function takes a binary value image as input and outputs the AABB of the object in the image which is indicated by its pixel value=255.
    """
    # find the indices of non-zero elements in the binary image
    non_zero_indices = np.nonzero(binary_image)
    if non_zero_indices[0].size == 0:
        return 0, 0, 0, 0

    # get the minimum and maximum x and y coordinates of the non-zero elements
    min_x = np.min(non_zero_indices[1])
    min_y = np.min(non_zero_indices[0])
    max_x = np.max(non_zero_indices[1])
    max_y = np.max(non_zero_indices[0])

    # return the AABB
    return (min_x, min_y, max_x, max_y)


def NormalizeImgToChanneled_CvFormat(m: cv.Mat):
    return m if len(m.shape) == 3 else m.reshape(m.shape + (1,))


def SeperateObject(m: np.ndarray):
    contours = cv.findContours(
        m.astype("uint8"), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )[0]
    ret: list[np.ndarray] = []
    for c in range(len(contours)):
        mcontour = np.zeros_like(m)
        cv.drawContours(mcontour, contours, c, 1, thickness=cv.FILLED)
        ret.append(mcontour)
    return ret


@dataclasses.dataclass
class MotionEstimator:
    mask: np.ndarray = None
    # subsample to be faster
    subsamplerate: float = 0.2
    # buffer fields
    lastScreen: np.ndarray = dataclasses.field(init=False, default=None)

    def cameramotion(self, newScreen, subsamplerate=1):
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        prev_pts = cv.goodFeaturesToTrack(
            self.lastScreen,
            maxCorners=100,
            qualityLevel=0.001,
            minDistance=3,
            blockSize=3,
            mask=self.mask,
        )
        if prev_pts is None:
            return [], identity

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv.calcOpticalFlowPyrLK(
            self.lastScreen, newScreen, prev_pts, None
        )

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        if idx.size == 0:
            return curr_pts, identity
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        prev_pts = prev_pts / subsamplerate
        curr_pts = curr_pts / subsamplerate

        # Find transformation matrix
        # m = cv.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)
        # will only work with OpenCV-3 or less
        m = cv.estimateAffinePartial2D(prev_pts, curr_pts, False)[0]

        # Extract traslation

        return curr_pts, m

    def resizeScreen(self, scr):
        return cv.resize(
            scr,
            None,
            fx=self.subsamplerate,
            fy=self.subsamplerate,
            interpolation=cv.INTER_AREA,
        )

    def __post_init__(self):
        if self.mask is not None:
            self.mask = self.resizeScreen(self.mask)

    def update(self, newScreen):
        newScreen = self.resizeScreen(newScreen)
        if self.lastScreen is None:
            self.lastScreen = newScreen
            return None
        else:
            ret = self.cameramotion(newScreen, self.subsamplerate)
            self.lastScreen = newScreen
            return ret


class AffineMats:
    zoom = lambda rate: np.array(
        [[rate, 0, 0], [0, rate, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    shift = lambda x, y: np.array(
        [[1, 0, x], [0, 1, y], [0, 0, 1]],
        dtype=np.float32,
    )
    flip = lambda lr, ud: np.array(
        [[lr, 0, 0], [0, ud, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    rot = lambda the: np.array(
        [[np.cos(the), np.sin(the), 0], [-np.sin(the), np.cos(the), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    identity = lambda: np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )


class MtiFilter:
    @dataclasses.dataclass
    class MtiFrame:
        img: np.ndarray

        # compared with prev frame
        cammotion: np.ndarray

    def __init__(self, mtiQueueSize, filter=None, camstablize=True) -> None:
        """
        consider storage only the transformed and meaned pic
        like the dynamic window way. kick the oldest one in queue out, and take its effect out of meaned pic
        """
        if filter is None:
            self.filter = interpolate.interp1d(
                [0, 0.3, 0.6, 1],
                [0, 0, 1, 1],
                kind="linear",
                bounds_error=False,
                fill_value=0,
                assume_sorted=True,
            )
        else:
            self.filter = filter
        self.camstablize = camstablize
        # fake type notation in order to scam ide type analysis
        self.mtiQueue: list[MtiFilter.MtiFrame] | AccessibleQueue = AccessibleQueue(
            mtiQueueSize
        )

        # try convienient type annotation but wont work
        # self.mtiQueue: AccessibleQueue.Annotation(
        #     MtiFilter.MtiStorage
        # ) = AccessibleQueue(5)

    def update(
        self, img: np.ndarray, roi: np.ndarray = None, cammotion: np.ndarray = None
    ):
        if not self.camstablize or cammotion is None:
            cammotion = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        if roi is None:
            cutRoiShift = AffineMats.identity()
            desiredImgShape = np.flip(img.shape[:2])
        else:
            cutRoiShift = AffineMats.shift(-roi[0], -roi[1])
            desiredImgShape = np.flip(np.array(roi[2:]) - np.array(roi[:2]))

        if self.mtiQueue.isEmpty():
            ret = None
        else:
            # mit proc
            def cammotionmat2x3to3x3(cammot: np.ndarray):
                return np.concatenate(
                    [
                        cammot,
                        [[0, 0, 1]],
                    ]
                )

            def cammotionmat3x3to2x3(cammot: np.ndarray):
                return cammot[:2, :]

            if self.camstablize:
                motionProd = cammotionmat2x3to3x3(cammotion)
            else:
                motionProd = None
            prevScreenAtNowViewList = []
            for i in range(len(self.mtiQueue)):
                # iter from the newest to oldest
                if self.camstablize:
                    # cut roi is done in affining
                    # warpAffine processing float32 is faster than uint8
                    prevScreenAtNowView = cv.warpAffine(
                        self.mtiQueue[-i].img,
                        cammotionmat3x3to2x3(cutRoiShift @ motionProd),
                        desiredImgShape,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    motionProd = (
                        cammotionmat2x3to3x3(self.mtiQueue[-i].cammotion) @ motionProd
                    )
                else:
                    if roi is not None:
                        # cut manually
                        # slightly faster, not significantly
                        prevScreenAtNowView = self.mtiQueue[-i].img[
                            roi[1] : roi[3], roi[0] : roi[2]
                        ]
                prevScreenAtNowViewList.append(prevScreenAtNowView)
            prevsignal = np.array(prevScreenAtNowViewList)
            delta = np.max(
                np.max(prevsignal, axis=0) - np.min(prevsignal, axis=0), axis=-1
            )
            # delta = np.max(np.std(prevsignal, axis=0), axis=-1)
            ret = self.filter(delta)
        self.mtiQueue.push__pop_if_full(MtiFilter.MtiFrame(img, cammotion))
        return ret
