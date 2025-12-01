import numpy as np

from endo_pipeline.library.analyze.kramersmoyal._km_computation import get_cartesian_product


def test_get_cartesian_product_2d():
    array = [np.array([1, 2, 3]), np.array([4, 5])]
    expected = np.array([[[1, 4], [1, 5]], [[2, 4], [2, 5]], [[3, 4], [3, 5]]])

    product = get_cartesian_product(array)

    assert (product == expected).all()


def test_get_cartesian_product_3d():
    array = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9])]
    expected = np.array(
        [
            [
                [[1, 4, 8], [1, 4, 9]],
                [[1, 5, 8], [1, 5, 9]],
                [[1, 6, 8], [1, 6, 9]],
                [[1, 7, 8], [1, 7, 9]],
            ],
            [
                [[2, 4, 8], [2, 4, 9]],
                [[2, 5, 8], [2, 5, 9]],
                [[2, 6, 8], [2, 6, 9]],
                [[2, 7, 8], [2, 7, 9]],
            ],
            [
                [[3, 4, 8], [3, 4, 9]],
                [[3, 5, 8], [3, 5, 9]],
                [[3, 6, 8], [3, 6, 9]],
                [[3, 7, 8], [3, 7, 9]],
            ],
        ]
    )

    product = get_cartesian_product(array)

    assert (product == expected).all()
