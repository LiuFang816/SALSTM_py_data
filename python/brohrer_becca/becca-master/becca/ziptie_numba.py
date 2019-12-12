"""
A set of functions written to help zipties take advantage of numba.

After a little experimentation, it appears that numba works best
when everything is written out explicitly in loops. It even beats
numpy handily on dense matrices. It doesn't like lists or lists of lists.
It also tends to work better if you
don't use numpy functions and if you make your code as simple
as possible.

The (nopython=True) call makes it so that if numba can't compile the code
to C (very fast), but is forced to fall back to python instead (dead slow
when doing loops), the function will fail and throw an error.
"""
from numba import jit


@jit(nopython=True)
def set_dense_val(array2d, i_rows, i_cols, val):
    """
    Set values in a dense 2D array using a list of indices.

    Parameters
    ----------
    array2d : 2D array of floats
        The array in which to set values.
    i_rows, i_cols: array of ints
        The row and column indices of each element to change.
    val : float
        The new value to assign.

    Returns
    -------
    Occur indirectly by modifying array2d.
    """
    for i, _ in enumerate(i_rows):
        array2d[i_rows[i], i_cols[i]] = val


@jit(nopython=True)
def max_dense(array2d, results):
    """
    Find the maximum value of a dense 2D array, with its row and column

    Parameters
    ----------
    array2d : 2D array of floats
        The array to find the maximum value of.
    results : array of floats, size 3
        An array for holding the results of the operation.

    Returns
    -------
    Results are returned indirectly by modifying results.
    The results array has three elements and holds
        [0] the maximum value found
        [1] the row number in which it was found
        [2] the column number in which it was found
    """
    max_val = results[0]
    i_row_max = results[1]
    i_col_max = results[2]
    for i_row in range(array2d.shape[0]):
        for i_col in range(array2d.shape[1]):
            if array2d[i_row, i_col] > max_val:
                max_val = array2d[i_row, i_col]
                i_row_max = i_row
                i_col_max = i_col
    results[0] = max_val
    results[1] = i_row_max
    results[2] = i_col_max


@jit(nopython=True)
def find_bundle_activities(i_rows, i_cols, cables, bundles, weights, threshold):
    """
    Use a greedy method to sparsely translate cables to bundles.

    Start at the last bundle added and work backward to the first.
    For each, calculate the bundle activity by finding the minimum
    value of each of its constituent cables. Then subtract out
    the bundle activity from each of its cables.

    Parameters
    ----------
    bundles : 1D array of floats
        An array of bundle activity values. Initially it is all zeros.
    cables : 1D array of floats
        An array of cable activity values.
    i_rows : array of ints
        The row indices for the non-zero sparse 2D array..
    i_cols : array of ints
        The column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
        Each column represents a cable and each row represents a bundle.
        The 2D array is a map from cables to bundles.
    threshold : float
        The amount of bundle activity below which, we just don't care.
    weights : array of floats
        A multiplier for how strongly the activity of each bundle should
        be considered when greedily selecting the next one to activate.

    Results
    -------
    Returned indirectly by modifying `cables. These are the residual
    cable activities that are not represented by any bundle activities.
    """
    large = 1e10
    max_vote = large

    # Repeat this process until the residual cable activities don't match
    # any bundles well.
    while max_vote > threshold:
        # Initialize the loop that greedily looks for the most strongly
        # activated bundle.
        max_vote = 0.
        best_val = 0.
        best_bundle = 0
        # This is the index in i_row and i_col where the
        # current bundle's cable constituents are listed. Cable indices
        # are assumed to be contiguous and bundles are assumed to be
        # listed in ascending order of index.
        i_best_bundle = 0

        # Iterate over each bundle, that is, over each row.
        i = len(i_rows) - 1
        row = i_rows[i]
        while row > -1:

            # For each bundle, find the minimum cable activity that
            # contribues to it.
            min_val = large
            n_cables = 0.
            i_bundle = i
            while i_rows[i] == row and i > -1:
                col = i_cols[i]
                val = cables[col]
                if val < min_val:
                    min_val = val
                n_cables += 1.
                i -= 1

            # The strength of the vote for the bundle is the minimum cable
            # activity multiplied by the number of cables. This weights
            # bundles with many member cables more highly than bundles
            # with few cables. It is a way to encourage sparsity and to
            # avoid creating more bundles than necessary.
            vote = min_val * (1. + .1 * (n_cables - 1.)) * (1. + weights[row])

            # Update the winning bundle if appropriate.
            if vote > max_vote:
                max_vote = vote
                best_val = min_val
                i_best_bundle = i_bundle
                best_bundle = row

            # Move on to the next bundle.
            row -= 1

        if best_val > 0.:
            # Set the bundle activity.
            bundles[best_bundle] = best_val

            # Subtract the bundle activity from each of the cables.
            # Using i_best_bundle lets us jump right to the place in
            # the list of indices where the cables for the winning bundle
            # are listed.
            i = i_best_bundle
            while i_rows[i] == best_bundle and i > -1:
                col = i_cols[i]
                cables[col] -= best_val
                i -= 1


@jit(nopython=True)
def nucleation_energy_gather(nonbundle_activities, nucleation_energy):
    """
    Gather nucleation energy.

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like
        nucleation_energy += (nonbundle_activities *
                              nonbundle_activities.T *
                              nucleation_energy_rate)
    Parameters
    ----------
    nonbundle_activities : array of floats
        The current activity of each input feature that is not explained by
        or captured in a bundle.
    nucleation_energy : 2D array of floats
        The amount of nucleation energy accumulated between each pair of
        input features.

    Results
    -------
    Returned indirectly by modifying nucleation_energy.
    """
    for i_cable1, _ in enumerate(nonbundle_activities):
        activity1 = nonbundle_activities[i_cable1]
        if activity1 > 0.:
            for i_cable2, _ in enumerate(nonbundle_activities):
                activity2 = nonbundle_activities[i_cable2]
                if activity2 > 0.:
                    nucleation_energy[i_cable1, i_cable2] += (
                        activity1 * activity2)


@jit(nopython=True)
def agglomeration_energy_gather(bundle_activities, nonbundle_activities,
                                n_bundles, agglomeration_energy):
    """
    Accumulate the energy binding a new feature to an existing bundle..

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like
        coactivities = bundle_activities * nonbundle_activities.T
        agglomeration_energy += coactivities * agglomeration_energy_rate

    Parameters
    ----------
    bundle_activities : array of floats
        The activity level of each bundle.
    nonbundle_activities : array of floats
        The current activity of each input feature that is not explained by
        or captured in a bundle.
    n_bundles : int
        The number of bundles that have been created so far.
    agglomeration_energy : 2D array of floats
        The total energy that has been accumulated between each input feature
        and each bundle.

    Results
    -------
    Returned indirectly by modifying `agglomeration_energy.
    """
    for i_col, _ in enumerate(nonbundle_activities):
        activity = nonbundle_activities[i_col]
        if activity > 0.:
            # Only decay bundles that have been created
            for i_row in range(n_bundles):
                if bundle_activities[i_row] > 0.:
                    coactivity = activity * bundle_activities[i_row]
                    agglomeration_energy[i_row, i_col] += coactivity
