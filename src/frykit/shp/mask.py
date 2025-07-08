from __future__ import annotations

from typing import cast

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray

from frykit.calc import asarrays, is_monotonic_decreasing, is_monotonic_increasing
from frykit.shp.typing import PolygonType
from frykit.utils import deprecator, format_type_error

__all__ = ["polygon_mask", "polygon_mask2", "polygon_to_mask"]


def polygon_mask(
    polygon: PolygonType, x: ArrayLike, y: ArrayLike, include_boundary: bool = True
) -> NDArray:
    """
    用多边形制作掩膜（mask）数组

    掩膜数组元素为 True 表示对应的数据点落入多边形内部，False 表示在外部。
    含 nan 或 inf 坐标的点直接返回 False。

    Parameters
    ----------
    polygon : PolygonType
        多边形对象

    x, y : array_like
        数据点的 xy 坐标。要求形状相同

    include_boundary : bool, default True
        是否包含落在多边形边界上的点。默认为 True。

    Returns
    -------
    mask : ndarray
        布尔类型的掩膜数组，形状与 x 和 y 相同。

    See Also
    --------
    - shapely.contains_xy
    - shapely.vectorized.contains
    """
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(
            format_type_error(
                "polygon", polygon, [shapely.Polygon, shapely.MultiPolygon]
            )
        )

    shapely.prepare(polygon)
    if include_boundary:
        predicate = shapely.covers
    else:
        predicate = shapely.contains_properly

    x, y = asarrays(x, y)
    if x.shape != y.shape:
        raise ValueError

    def do_recursion(x: NDArray, y: NDArray) -> NDArray:
        if len(x) == 0:
            return np.array([], dtype=bool)

        # 只有一个元素时无惧浮点误差
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        x_overlapping = x0 == x1
        y_overlapping = y0 == y1
        mask = np.zeros_like(x, dtype=bool)

        # 处理数据点重合成单点或线段的情况
        if x_overlapping and y_overlapping:
            mask[:] = predicate(polygon, shapely.Point(x0, y0))
            return mask
        elif x_overlapping or y_overlapping:
            geometry = shapely.LineString([(x0, y0), (x1, y1)])
        else:
            geometry = shapely.box(x0, y0, x1, y1)

        if polygon.disjoint(geometry):
            return mask
        if predicate(polygon, geometry):
            mask[:] = True
            return mask

        # 递归切割矩形
        xm = x0 + (x1 - x0) / 2
        ym = y0 + (y1 - y0) / 2
        lower = y <= ym
        upper = ~lower
        left = x <= xm
        right = ~left
        for m in [lower & left, lower & right, upper & left, upper & right]:
            mask[m] = do_recursion(x[m], y[m])

        return mask

    # 处理 nan 和 inf
    valid = np.isfinite(x) & np.isfinite(y)
    mask = np.zeros_like(x, dtype=bool)
    mask[valid] = do_recursion(x[valid], y[valid])
    if mask.ndim == 0:
        mask = np.bool_(mask)

    return cast(NDArray, mask)


# https://gist.github.com/perrette/a78f99b76aed54b6babf3597e0b331f8
def polygon_mask2(
    polygon: PolygonType, x: ArrayLike, y: ArrayLike, include_boundary: bool = True
) -> NDArray:
    """
    用多边形制作掩膜（mask）数组

    掩膜数组元素为 True 表示对应的网格点落入多边形内部，False 表示在外部。

    与 polygon_mask 函数的区别是要求 x 和 y 是描述二维直线网格的一维坐标，
    当点数很多时计算速度更快。

    Parameters
    ----------
    polygon : PolygonType
        多边形对象

    x : (nx,) array_like
        网格的 x 坐标。要求单调递增或递减。

    y : (ny,) array_like
        网格的 y 坐标。要求单调递增或递减。

    include_boundary : bool, default True
        是否包含落在多边形边界上的点。默认为 True。

    Returns
    -------
    mask : (ny, nx) ndarray
        布尔类型的掩膜数组

    See Also
    --------
    - shapely.contains_xy
    - shapely.vectorized.contains
    """
    if not isinstance(polygon, (shapely.Polygon, shapely.MultiPolygon)):
        raise TypeError(
            format_type_error(
                "polygon", polygon, [shapely.Polygon, shapely.MultiPolygon]
            )
        )

    shapely.prepare(polygon)
    if include_boundary:
        predicate = shapely.covers
    else:
        predicate = shapely.contains_properly

    x, y = asarrays(x, y)
    if x.ndim != 1:
        raise ValueError("x 必须是一维数组")
    if y.ndim != 1:
        raise ValueError("y 必须是一维数组")

    # 保证 x 升序
    if is_monotonic_increasing(x):
        x_ascending = True
    elif is_monotonic_decreasing(x):
        x_ascending = False
        x = x[::-1]
    else:
        raise ValueError("要求 x 单调递增或递减")

    # 保证 y 升序
    if is_monotonic_increasing(y):
        y_ascending = True
    elif is_monotonic_decreasing(y):
        y_ascending = False
        y = y[::-1]
    else:
        raise ValueError("要求 y 单调递增或递减")

    def do_recursion(x: NDArray, y: NDArray, mask: NDArray) -> None:
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return

        x0, x1 = x[0], x[-1]
        y0, y1 = y[0], y[-1]
        x_overlapping = x0 == x1
        y_overlapping = y0 == y1

        if x_overlapping and y_overlapping:
            mask[:] = predicate(polygon, shapely.Point(x0, y0))
            return
        elif x_overlapping or y_overlapping:
            geometry = shapely.LineString([(x0, y0), (x1, y1)])
        else:
            geometry = shapely.box(x0, y0, x1, y1)

        if polygon.disjoint(geometry):
            return
        if predicate(polygon, geometry):
            mask[:] = True
            return

        hx, hy = nx // 2, ny // 2
        do_recursion(x[:hx], y[:hy], mask[:hy, :hx])
        do_recursion(x[:hx], y[hy:], mask[hy:, :hx])
        do_recursion(x[hx:], y[:hy], mask[:hy, hx:])
        do_recursion(x[hx:], y[hy:], mask[hy:, hx:])

    # 恢复顺序
    mask = np.zeros((len(y), len(x)), dtype=bool)
    do_recursion(x, y, mask)
    if not x_ascending:
        mask = mask[:, ::-1]
    if not y_ascending:
        mask = mask[::-1, :]

    return mask


@deprecator(alternative="frykit.shp.polygon_mask")
def polygon_to_mask(polygon: PolygonType, x: ArrayLike, y: ArrayLike) -> NDArray:
    return polygon_mask(polygon, x, y, include_boundary=False)
