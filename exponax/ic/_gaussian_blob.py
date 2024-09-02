from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from ._base_ic import BaseIC, BaseRandomICGenerator

D = TypeVar("D")


class GaussianBlob(eqx.Module):
    position: Float[Array, "D"]
    covariance: Float[Array, "D D"]
    _inv_covariance: Float[Array, "D D"]
    one_complement: bool

    def __init__(
        self,
        position: Float[Array, "D"],
        covariance: Float[Array, "D D"],
        *,
        one_complement: bool = False,
    ):
        """
        A state described by a Gaussian blob. Note that the produced function is
        not perfectly periodic, especially if the blobs are close to the domain
        boundaries.

        **Arguments**:

        - `position`: The position of the blob.
        - `covariance`: The covariance matrix of the blob.
        - `one_complement`: Whether to return one minus the Gaussian blob.
        """
        self.position = position
        self.covariance = covariance
        self._inv_covariance = jnp.linalg.inv(covariance)
        self.one_complement = one_complement

    def __call__(self, x: Float[Array, "D ... N"]) -> Float[Array, "1 ... N"]:
        num_spatial_dims = x.shape[0]
        if num_spatial_dims != self.position.shape[0]:
            raise ValueError(
                f"Expected {self.position.shape[0]} spatial dimensions, but got {num_spatial_dims}."
            )

        position_expanded = jnp.reshape(
            self.position, (self.position.shape[0],) + (1,) * num_spatial_dims
        )

        diff = x - position_expanded

        quadratic_form = jnp.einsum(
            "i...,ij,j...->...",
            diff,
            self._inv_covariance,
            diff,
        )

        blob = jnp.exp(-0.5 * quadratic_form)

        # Add back singleton channel axis
        blob = jnp.expand_dims(blob, axis=0)

        if self.one_complement:
            return 1.0 - blob
        else:
            return blob


class GaussianBlobs(BaseIC):
    blob_list: tuple[GaussianBlob, ...]

    def __init__(
        self,
        blob_list: tuple[GaussianBlob, ...],
    ):
        """
        A state described by a collection of Gaussian blobs.

        **Arguments**:

        - `blob_list`: A tuple of Gaussian blobs.
        """
        self.blob_list = blob_list

    def __call__(self, x: Array) -> Array:
        summation = sum(blob(x) for blob in self.blob_list)
        return summation / len(self.blob_list)


class RandomGaussianBlobs(BaseRandomICGenerator):
    num_spatial_dims: int
    domain_extent: float
    num_blobs: int

    position_range: tuple[float, float]
    variance_range: tuple[float, float]

    one_complement: bool

    def __init__(
        self,
        num_spatial_dims: int,
        *,
        domain_extent: float = 1.0,
        num_blobs: int = 1,
        position_range: tuple[float, float] = (0.4, 0.6),
        variance_range: tuple[float, float] = (0.005, 0.01),
        one_complement: bool = False,
    ):
        """
        A random Gaussian blob initial condition generator.

        **Arguments**:

        - `num_spatial_dims`: The number of spatial dimensions.
        - `domain_extent`: The extent of the domain.
        - `num_blobs`: The number of blobs.
        - `position_range`: The range of the position of the blobs. This
            will be scaled by the domain extent. Hence, this acts as if the
            domain_extent was 1
        - `variance_range`: The range of the variance of the blobs. This will
            be scaled by the domain extent. Hence, this acts as if the
            domain_extent was 1
        - `one_complement`: Whether to return one minus the Gaussian blob.
        """
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_blobs = num_blobs
        self.position_range = position_range
        self.variance_range = variance_range
        self.one_complement = one_complement

    def gen_blob(self, *, key) -> GaussianBlob:
        """
        Generates a single Gaussian blob.
        """
        position_key, variance_key = jr.split(key)

        position = jr.uniform(
            position_key,
            shape=(self.num_spatial_dims,),
            minval=self.position_range[0] * self.domain_extent,
            maxval=self.position_range[1] * self.domain_extent,
        )
        variances = jr.uniform(
            variance_key,
            shape=(self.num_spatial_dims,),
            minval=self.variance_range[0] * self.domain_extent,
            maxval=self.variance_range[1] * self.domain_extent,
        )
        covariance = jnp.diag(variances)

        return GaussianBlob(position, covariance, one_complement=self.one_complement)

    def gen_ic_fun(self, *, key: PRNGKeyArray) -> GaussianBlobs:
        blob_list = []
        for _ in range(self.num_blobs):
            key, subkey = jr.split(key)
            blob_list.append(self.gen_blob(key=subkey))
        return GaussianBlobs(tuple(blob_list))
