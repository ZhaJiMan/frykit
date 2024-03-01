from collections.abc import Sequence
from typing import Any, Optional, Union

from PIL import Image
import numpy as np

from frykit.help import PathType


# TODO: 支持输入为Image的序列.
def make_gif(
    img_filepaths: Sequence[PathType],
    gif_filepath: PathType,
    duration: int = 500,
    loop: int = 0,
    optimize: bool = False,
) -> None:
    '''
    制作GIF图. 结果的mode和尺寸由第一张图决定.

    Parameters
    ----------
    img_filepaths : sequence of PathType
        图片路径的列表.

    gif_filepath : PathType
        输出GIF图片的路径.

    duration : int or list or tuple, optional
        每一帧的持续时间, 单位为毫秒. 也可以用列表或元组分别指定每一帧的持续时间.
        默认为500ms=0.5s.

    loop : int, optional
        GIF图片循环播放的次数. 默认无限循环.

    optimize : bool, optional
        尝试压缩GIF图片的调色板.
    '''
    if not img_filepaths:
        raise ValueError('至少需要一张图片')

    images = [Image.open(str(filepath)) for filepath in img_filepaths]
    images[0].save(
        str(gif_filepath),
        format='gif',
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=optimize,
    )


# TODO: 加入从右往左的粘贴顺序?
def merge_images(
    filepaths: Sequence[PathType],
    shape: Optional[tuple[int, int]] = None,
    mode: Optional[str] = None,
    bgcolor: Any = 'white',
) -> Image.Image:
    '''
    合并一组图片.

    合并结果可以分为shape[0]行和shape[1]列个格子. 将图片从左到右, 从上到下
    依次填入格子中, 格子大小由这组图片的最大宽高决定. 当图片比格子小时居中摆放.
    shape可以含有-1, 表示行或列数自动根据图片数计算得到.

    Parameters
    ----------
    filepaths : sequence of PathType
        图片路径的列表.

    shape : (2,) tuple of int, optional
        合并的行列形状. 默认为None, 表示纵向拼接.

    mode : str, optional
        合并后图片的mode. 默认为None, 表示采用第一张图片的mode.

    bgcolor : optional
        合并后图片的背景颜色. 默认为白色.

    Returns
    -------
    merged : Image
        合并后的图片.
    '''
    if not filepaths:
        raise ValueError('至少需要一张图片')

    # 获取最大宽高.
    images = []
    width = height = 0
    for filepath in filepaths:
        image = Image.open(str(filepath))
        images.append(image)
        if image.width > width:
            width = image.width
        if image.height > height:
            height = image.height

    # 决定行列.
    num = len(images)
    if shape is None:
        shape = (num, 1)
    row, col = shape
    if row == 0 or row < -1 or col == 0 or col < -1:
        raise ValueError('shape的元素只能为正数或-1')
    if row == -1 and col == -1:
        raise ValueError('shape的元素不能同时为-1')
    if row == -1:
        row = (num - 1) // col + 1
    if col == -1:
        col = (num - 1) // row + 1

    # 粘贴到画布上.
    merged = Image.new(
        mode=images[0].mode if mode is None else mode,
        size=(col * width, row * height),
        color=bgcolor,
    )
    for k, image in enumerate(images):
        i, j = divmod(k, col)
        left = j * width + (width - image.width) // 2
        top = i * height + (height - image.height) // 2
        merged.paste(image, (left, top))

    return merged


def split_image(
    filepath: PathType, shape: Union[int, tuple[int, int]]
) -> list[Image.Image]:
    '''
    分割一张图片.

    将图片分割成shape[0]行和shape[1]列, 然后按从左往右从上往下的顺序收集结果.

    Parameters
    ----------
    filepath : PathType
        图片路径

    shape : (2,) int or tuple of int
        分割的行列形状. 整数表示仅按行分割.

    Returns
    -------
    images : list of Image
        分割出来的一组图片.
    '''
    row, col = (shape, 1) if isinstance(shape, int) else shape
    if row <= 0 or col <= 0:
        raise ValueError('shape只能含正数维度')

    # 可能无法整除.
    image = Image.open(str(filepath))
    width = image.width // col
    height = image.height // row

    images = []
    for i in range(row):
        for j in range(col):
            left = j * width
            right = left + width
            top = i * height
            lower = top + height
            part = image.crop((left, top, right, lower))
            images.append(part)

    return images


def compare_images(filepath1: PathType, filepath2: PathType) -> Image.Image:
    '''
    通过求两张图片的绝对差值比较前后差异.

    要求两张图片大小和模式相同.

    Parameters
    ----------
    filepath1 : PathType
        第一张图片的路径.

    filepath2 : PathType
        第二张图片的路径.

    Returns
    -------
    images : list of Image
        两张图片的绝对差值构成的图片.
    '''
    image1 = Image.open(str(filepath1))
    image2 = Image.open(str(filepath2))
    if image1.size != image2.size:
        raise ValueError('两张图片的宽高不同')
    if image1.mode != image2.mode:
        raise ValueError('两张图片的mode不同')
    arr1 = np.asarray(image1, np.int16)
    arr2 = np.asarray(image2, np.int16)
    diff = np.abs(arr1 - arr2).astype(np.uint8)
    diff = Image.fromarray(diff)

    return diff
