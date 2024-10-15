import numpy as np

import exponax as ex


def test_low_pass_filter_masks_1d():
    # Need to test both for even and odd number of points because that changes
    # how the Nyquist mode is treated.
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(1, 10, cutoff=3),
        np.array([[True, True, True, True, False, False]]),
    )
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(1, 11, cutoff=3),
        np.array([[True, True, True, True, False, False]]),
    )


def test_nyquist_filter_masks_1d():
    np.testing.assert_equal(
        ex.spectral.oddball_filter_mask(1, 10),
        np.array([[True, True, True, True, True, False]]),
    )
    np.testing.assert_equal(
        ex.spectral.oddball_filter_mask(1, 11),
        np.array([[True, True, True, True, True, True]]),
    )


def test_low_pass_filter_masks_2d():
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(2, 10, cutoff=3),
        np.array(
            [
                [
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                ]
            ]
        ),
    )
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(2, 11, cutoff=3),
        np.array(
            [
                [
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                ]
            ]
        ),
    )
    # Below is with `axis_separate=False` which not creates `True`-hypercube
    # regions but spheres (in 3d) or circles (in 2d) of `True` values.
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(2, 10, cutoff=3, axis_separate=False),
        np.array(
            [
                [
                    [True, True, True, True, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, False, False, False],
                    [True, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [True, False, False, False, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, False, False, False],
                ]
            ]
        ),
    )
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(2, 11, cutoff=3, axis_separate=False),
        np.array(
            [
                [
                    [True, True, True, True, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, False, False, False],
                    [True, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [True, False, False, False, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, False, False, False],
                ]
            ]
        ),
    )


def test_nyquist_filter_masks_2d():
    np.testing.assert_equal(
        ex.spectral.oddball_filter_mask(2, 10),
        np.array(
            [
                [
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, False],
                    [False, False, False, False, False, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, False],
                ]
            ]
        ),
    )
    np.testing.assert_equal(
        ex.spectral.oddball_filter_mask(2, 11),
        np.array(
            [
                [
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                ]
            ]
        ),
    )


def test_low_pass_filter_masks_3d():
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(3, 8, cutoff=2),
        np.array(
            [
                [
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                ]
            ]
        ),
    )
    np.testing.assert_equal(
        ex.spectral.low_pass_filter_mask(3, 9, cutoff=2),
        np.array(
            [
                [
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                ]
            ]
        ),
    )

    # TODO: Add tests for `axis_separate=False` in 3D.


def test_nyquist_filter_masks_3d():
    np.testing.assert_equal(
        ex.spectral.oddball_filter_mask(3, 8),
        np.array(
            [
                [
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [False, False, False, False, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                    ],
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [False, False, False, False, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                    ],
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [False, False, False, False, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                    ],
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [False, False, False, False, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                    ],
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ],
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [False, False, False, False, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                    ],
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [False, False, False, False, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                    ],
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [False, False, False, False, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                        [True, True, True, True, False],
                    ],
                ]
            ]
        ),
    )

    np.testing.assert_equal(
        ex.spectral.oddball_filter_mask(3, 9),
        np.ones((1, 9, 9, (9 // 2) + 1), dtype=bool),
    )
