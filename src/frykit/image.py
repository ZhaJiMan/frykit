from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from frykit.typing import PathType

ImageInput: TypeAlias = PathType | Image.Image


def _read_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    with Image.open(image) as im:
        im.load()
        return im


# TODO: alpha
def make_gif(images: Sequence[ImageInput], filepath: PathType, **kwargs: Any) -> None:
    """
    制作 gif 图。结果的 mode 和尺寸由第一张图决定。

    Parameters
    ----------
    images : sequence of ImageInput
        输入的一组图片

    filepath : PathType
        输出 gif 图片的路径

    **kwargs
        用 pillow 保存 gif 时的参数。
        例如 duration、loop、disposal、transparency 等。
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    """
    if len(images) == 0:
        raise ValueError("至少需要一张图片")

    kwargs.setdefault("loop", 0)
    kwargs.setdefault("duration", 500)

    images = list(map(_read_image, images))
    images = cast(list[Image.Image], images)

    images[0].save(
        filepath, format="gif", save_all=True, append_images=images[1:], **kwargs
    )


def merge_images(
    images: Any,  # TODO: type hint
    mode: str | None = None,
    bgcolor: float | tuple[float, ...] | str | None = "white",
) -> Image.Image:
    """
    合并一组图片

    Parameters
    ----------
    images : array_like of ImageInput
        输入的一组图片。用 object 类型的数组表示，其二维排布决定了合并结果的行列。
        合并行列的各自大小由 images 里的最大宽高决定，当图片比格子小时居中摆放。
        数组可以含有 None，表示占位的空白图片。

    mode : str or None, default None
        合并后图片的 mode。默认为 None，表示采用第一张图片的 mode。

    bgcolor : float, tuple of float, str or None, default 'white'
        合并后图片的背景颜色。默认为白色。

    Returns
    -------
    merged : Image
        合并后的图片
    """
    images = np.array(images, dtype=object)
    images = np.atleast_2d(images)
    if images.ndim > 2:
        raise ValueError("images 的维度不能超过 2")

    max_width = 0
    max_height = 0
    first_image = None
    for index in np.ndindex(images.shape):
        if images[index] is None:
            continue
        image = _read_image(images[index])
        images[index] = image
        if first_image is None:
            first_image = image
        max_width = max(max_width, image.width)
        max_height = max(max_height, image.height)

    if first_image is None:
        raise ValueError("images 为空或者全为 None")

    nrows, ncols = images.shape
    merged = Image.new(
        mode=first_image.mode if mode is None else mode,
        size=(ncols * max_width, nrows * max_height),
        color=bgcolor,
    )

    # 居中粘贴
    for (i, j), image in np.ndenumerate(images):
        if image is not None:
            left = j * max_width + (max_width - image.width) // 2
            top = i * max_height + (max_height - image.height) // 2
            merged.paste(image, (left, top))

    return merged


def split_image(image: ImageInput, shape: int | tuple[int, int]) -> NDArray:
    """
    将一张图片分割成形如 shape 的图片数组

    Parameters
    ----------
    image : ImageInput
        输入图片

    shape : int or (2,) tuple of int
        分割的行列形状。规则同 numpy。

    Returns
    -------
    split : ndarray of Image
        形如 shape 的图片数组
    """
    is_1d = isinstance(shape, int)
    nrows, ncols = (1, shape) if is_1d else shape
    if nrows <= 0 or ncols <= 0:
        raise ValueError("shape 只能含正整数维度")

    # 可能无法整除
    image = _read_image(image)
    width = image.width // ncols
    height = image.height // nrows

    split = np.empty((nrows, ncols), object)
    for i in range(nrows):
        for j in range(ncols):
            left = j * width
            right = left + width
            top = i * height
            lower = top + height
            split[i, j] = image.crop((left, top, right, lower))

    return split[0, :] if is_1d else split


def compare_images(image1: ImageInput, image2: ImageInput) -> Image.Image:
    """
    通过求两张图片的绝对差值比较前后差异

    要求两张图片大小和模式相同。

    Parameters
    ----------
    image1 : ImageInput
        第一张图片

    image2 : ImageInput
        第二张图片

    Returns
    -------
    images : list of Image
        两张图片的绝对差值构成的图片
    """
    image1 = _read_image(image1)
    image2 = _read_image(image2)
    if image1.size != image2.size:
        raise ValueError("两张图片的宽高不同")
    if image1.mode != image2.mode:
        raise ValueError("两张图片的 mode 不同")
    arr1 = np.asarray(image1, np.int16)
    arr2 = np.asarray(image2, np.int16)
    diff = np.abs(arr1 - arr2).astype(np.uint8)
    diff = Image.fromarray(diff)

    return diff
