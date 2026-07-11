from __future__ import annotations

import numpy as np
import pytest

from fjs.reconstruction import (
    orthonormalize_candidate_directions,
    replace_spectral_subspace,
)

pytestmark = pytest.mark.unit


def test_nonorthogonal_candidates_install_exact_subspace_eigenpairs() -> None:
    baseline = np.array(
        [[3.0, 0.4, 0.2], [0.4, 2.0, -0.1], [0.2, -0.1, 1.0]],
        dtype=np.float64,
    )
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    targets = np.array([5.0, 2.5], dtype=np.float64)
    basis = orthonormalize_candidate_directions(directions, dimension=3)

    reconstructed = replace_spectral_subspace(baseline, directions, targets)

    assert reconstructed @ basis == pytest.approx(
        basis @ np.diag(targets),
        abs=1e-10,
    )
    projector = np.eye(3) - basis @ basis.T
    assert projector @ reconstructed @ projector == pytest.approx(
        projector @ baseline @ projector,
        abs=1e-10,
    )


def test_multi_candidate_replacement_is_permutation_and_sign_invariant() -> None:
    baseline = np.array(
        [[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]],
        dtype=np.float64,
    )
    first = np.array([1.0, 0.0, 0.0])
    second = np.array([1.0, 2.0, 0.0])

    expected = replace_spectral_subspace(
        baseline,
        [first, second],
        [4.0, 2.0],
    )
    permuted = replace_spectral_subspace(
        baseline,
        [-second, first],
        [2.0, 4.0],
    )

    assert permuted == pytest.approx(expected, abs=1e-10)


def test_rank_deficient_candidate_set_fails_loudly() -> None:
    with pytest.raises(ValueError, match="rank-deficient"):
        replace_spectral_subspace(
            np.eye(3),
            [np.array([1.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0])],
            [3.0, 2.0],
        )
