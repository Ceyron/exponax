import exponax as ex


def test_mode_slices_generation():
    # 1D
    assert ex.spectral.get_modes_slices(1, 10) == (
        (
            slice(None),
            slice(None, (10 // 2) + 1),
        ),
    )
    assert ex.spectral.get_modes_slices(1, 11) == (
        (
            slice(None),
            slice(None, (11 // 2) + 1),
        ),
    )

    # 2D
    assert ex.spectral.get_modes_slices(2, 10) == (
        (
            slice(None),
            slice(None, (10 // 2)),
            slice(None, (10 // 2) + 1),
        ),
        (
            slice(None),
            slice(-(10 // 2), None),
            slice(None, (10 // 2) + 1),
        ),
    )
    assert ex.spectral.get_modes_slices(2, 11) == (
        (
            slice(None),
            slice(None, (11 // 2) + 1),
            slice(None, (11 // 2) + 1),
        ),
        (
            slice(None),
            slice(-(11 // 2), None),
            slice(None, (11 // 2) + 1),
        ),
    )

    # 3D
    assert ex.spectral.get_modes_slices(3, 10) == (
        (
            slice(None),
            slice(None, (10 // 2)),
            slice(None, (10 // 2)),
            slice(None, (10 // 2) + 1),
        ),
        (
            slice(None),
            slice(-(10 // 2), None),
            slice(None, (10 // 2)),
            slice(None, (10 // 2) + 1),
        ),
        (
            slice(None),
            slice(None, (10 // 2)),
            slice(-(10 // 2), None),
            slice(None, (10 // 2) + 1),
        ),
        (
            slice(None),
            slice(-(10 // 2), None),
            slice(-(10 // 2), None),
            slice(None, (10 // 2) + 1),
        ),
    )
    assert ex.spectral.get_modes_slices(3, 11) == (
        (
            slice(None),
            slice(None, (11 // 2) + 1),
            slice(None, (11 // 2) + 1),
            slice(None, (11 // 2) + 1),
        ),
        (
            slice(None),
            slice(-(11 // 2), None),
            slice(None, (11 // 2) + 1),
            slice(None, (11 // 2) + 1),
        ),
        (
            slice(None),
            slice(None, (11 // 2) + 1),
            slice(-(11 // 2), None),
            slice(None, (11 // 2) + 1),
        ),
        (
            slice(None),
            slice(-(11 // 2), None),
            slice(-(11 // 2), None),
            slice(None, (11 // 2) + 1),
        ),
    )
