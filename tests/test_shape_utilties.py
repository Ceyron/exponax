import exponax as ex


def test_space_indices():
    assert ex.spectral.space_indices(1) == (-1,)
    assert ex.spectral.space_indices(2) == (-2, -1)
    assert ex.spectral.space_indices(3) == (-3, -2, -1)


def test_spatial_shape():
    assert ex.spectral.spatial_shape(1, 64) == (64,)
    assert ex.spectral.spatial_shape(2, 64) == (64, 64)
    assert ex.spectral.spatial_shape(3, 64) == (64, 64, 64)


def test_wavenumber_shape():
    assert ex.spectral.wavenumber_shape(1, 64) == (33,)
    assert ex.spectral.wavenumber_shape(2, 64) == (64, 33)
    assert ex.spectral.wavenumber_shape(3, 64) == (64, 64, 33)

    assert ex.spectral.wavenumber_shape(1, 65) == (33,)
    assert ex.spectral.wavenumber_shape(2, 65) == (65, 33)
    assert ex.spectral.wavenumber_shape(3, 65) == (65, 65, 33)
