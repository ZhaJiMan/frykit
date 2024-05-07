from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from frykit._typing import PathType

ImageInput = Union[PathType, Image.Image]


def _read_image(image: ImageInput) -> Image.Image:
    '''读取图片为Image对象.'''
    if isinstance(image, Image.Image):
        return image
    return Image.open(str(image))


def make_gif(
    images: Sequence[ImageInput], filepath: PathType, **kwargs: Any
) -> None:
    '''
    制作GIF图. 结果的mode和尺寸由第一张图决定.

    Parameters
    ----------
    images : sequence of ImageInput
        输入的一组图片.

    filepath : PathType
        输出GIF图片的路径.

    **kwargs
        用pillow保存GIF时的参数.
        例如duration, loop, disposal, transparency等.
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    '''
    if not images:
        raise ValueError('至少需要一张图片')

    kwargs.setdefault('duration', 500)
    kwargs.setdefault('loop', 0)

    images = [_read_image(image) for image in images]
    images[0].save(
        str(filepath),
        format='gif',
        save_all=True,
        append_images=images[1:],
        **kwargs
    )


def merge_images(
    images: Sequence[ImageInput],
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
    images : sequence of ImageInput
        输入的一组图片.

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
    if not images:
        raise ValueError('至少需要一张图片')

    # 获取最大宽高.
    images = [_read_image(image) for image in images]
    width = max(image.width for image in images)
    height = max(image.height for image in images)

    # 决定行列.
    num = len(images)
    if shape is None:
        shape = (num, 1)
    row, col = shape
    if row == 0 or row < -1 or col == 0 or col < -1:
        raise ValueError('shape的元素只能为正整数或-1')
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
    image: ImageInput, shape: Union[int, tuple[int, int]]
) -> np.ndarray:
    '''
    将一张图片分割成形如shape的图片数组.

    Parameters
    ----------
    image : ImageInput
        输入图片.

    shape : int or (2,) tuple of int
        分割的行列形状. 规则类似NumPy.

    Returns
    -------
    split : ndarray of Image
        形如shape的图片数组.
    '''
    is_1d = isinstance(shape, int)
    row, col = (1, shape) if is_1d else shape
    if row <= 0 or col <= 0:
        raise ValueError('shape只能含正整数维度')

    # 可能无法整除.
    image = _read_image(image)
    width = image.width // col
    height = image.height // row

    split = np.empty((row, col), object)
    for i in range(row):
        for j in range(col):
            left = j * width
            right = left + width
            top = i * height
            lower = top + height
            split[i, j] = image.crop((left, top, right, lower))

    return split[0, :] if is_1d else split


def compare_images(image1: ImageInput, image2: ImageInput) -> Image.Image:
    '''
    通过求两张图片的绝对差值比较前后差异.

    要求两张图片大小和模式相同.

    Parameters
    ----------
    image1 : ImageInput
        第一张图片.

    image2 : ImageInput
        第二张图片.

    Returns
    -------
    images : list of Image
        两张图片的绝对差值构成的图片.
    '''
    image1 = _read_image(image1)
    image2 = _read_image(image2)
    if image1.size != image2.size:
        raise ValueError('两张图片的宽高不同')
    if image1.mode != image2.mode:
        raise ValueError('两张图片的mode不同')
    arr1 = np.asarray(image1, np.int16)
    arr2 = np.asarray(image2, np.int16)
    diff = np.abs(arr1 - arr2).astype(np.uint8)
    diff = Image.fromarray(diff)

    return diff
