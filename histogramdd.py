import torch

_range = range
import numpy as np


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (torch.isfinite(first_edge) and torch.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (torch.isfinite(first_edge) and torch.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def histogramdd(thetas, rhos, rho_values, displacements=None, range=None):
    """
    Compute the bidimensional histogram of some data.
    Parameters
    ----------
    sample : (N, D) array, or (D, N) array_like
        The data to be histogrammed.
        Note the unusual interpretation of sample when an array_like:
        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramgramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramgramdd((X, Y, Z))``.
        The first form should be preferred.
    bins : sequence or int, optional
        The bin specification:
        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).
    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of D None values.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.
    normed : bool, optional
        An alias for the density argument that behaves identically. To avoid
        confusion with the broken normed argument to `histogram`, `density`
        should be preferred.
    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        Weights are normalized to 1 if normed is True. If normed is False,
        the values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.
    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See normed and weights
        for the different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.
    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram
    Examples
    --------
    >>> r = np.random.randn(100,3)
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)
    """
    bins = [thetas.float(), rhos.float()]
    sample = torch.stack([
        torch.tile(thetas, (rho_values.shape[0], 1)).ravel(),
        rho_values.ravel()
    ]).T
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = torch.atleast_2d(sample).T
        N, D = sample.shape

    nbin = torch.empty(D, dtype=torch.int64)
    edges = D*[None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]

    # normalize the range argument
    if range is None:
        range = (None,) * D
    elif len(range) != D:
        raise ValueError('range argument must have one entry per dimension')

    # Create edge arrays
    for i in _range(D):
        if bins[i].ndim == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            edges[i] = np.linspace(smin, smax, bins[i] + 1)
        elif bins[i].ndim == 1:
            edges[i] = torch.as_tensor(bins[i])
            if torch.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                    .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) # + 1  # includes an outlier on each end

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        torch.searchsorted(edges[i], sample[:, i], side='left')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    # for i in _range(D):
    #     # Find which points are on the rightmost edge.
    #     on_edge = (sample[:, i] == edges[i][-1])
    #     # Shift these points one bin to the right.
    #     Ncount[i][on_edge] -= 1

    Ncount = torch.stack(Ncount, dim=1)
    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = Ncount @ torch.tensor([nbin[1], 1])

    if displacements is not None:
        displacements_adapted = displacements.repeat(len(thetas)).reshape(len(thetas), len(displacements)).T.flatten()
        # Add the displacements
        xy_enlarged = torch.concat([torch.arange(-dis, dis + 1) + xyindex
                                    for dis, xyindex in zip(displacements_adapted, xy)])
    else:
        xy_enlarged = xy

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = torch.bincount(xy_enlarged, minlength=nbin.prod())

    # Shape into a proper matrix
    hist = hist.reshape(*nbin)

    # Remove outliers (indices 0 and -1 for each dimension).
    # core = D*(slice(1, -1),)
    # hist = hist[core]

    # if (torch.tensor(hist.shape) != nbin - 2).any():
    #     raise RuntimeError(
    #         "Internal Shape Error")
    return hist, edges