"""
The Ziptie class.
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import becca.tools as tools
import becca.ziptie_numba as nb


class Ziptie(object):
    """
    An incremental unsupervised clustering algorithm.

    Input channels are clustered together into mutually co-active sets.
    A helpful metaphor is bundling cables together with zip ties.
    Cables that carry related signals are commonly co-active,
    and are grouped together. Cables that are co-active with existing
    bundles can be added to those bundles. A single cable may be ziptied
    into several different bundles. Co-activity is estimated
    incrementally, that is, the algorithm updates the estimate after
    each new set of signals is received.

    When stacked with other levels,
    zipties form a sparse deep neural network (DNN).
    This DNN has the extremely desirable characteristic of
    l-0 sparsity--the number of non-zero weights are minimized.
    The vast majority of weights in this network are zero,
    and the rest are one.
    This makes sparse computation feasible and allows for
    straightforward interpretation and visualization of the
    features.
    """

    def __init__(self,
                 num_cables,
                 num_bundles=None,
                 name=None,
                 debug=False):
        """
        Initialize the ziptie, pre-allocating data structures.

        Parameters
        ----------
        debug : boolean, optional
            Indicate whether to print informative status messages
            during execution. Default is False.
        num_bundles : int, optional
            The number of bundle outputs from the Ziptie.
        num_cables : int
            The number of inputs to the Ziptie.
        name : str, optional
            The name assigned to the Ziptie.
            Default is 'ziptie'.
        """
        # name : str
        #     The name associated with the Ziptie.
        if name is None:
            self.name = 'ziptie'
        else:
            self.name = name
        # debug : boolean
        #     Indicate whether to print informative status messages
        #     during execution.
        self.debug = debug
        # max_num_cables : int
        #     The maximum number of cable inputs allowed.
        self.max_num_cables = num_cables
        # max_num_bundles : int
        #     The maximum number of bundle outputs allowed.
        if num_bundles is None:
            self.max_num_bundles = self.max_num_cables
        else:
            self.max_num_bundles = num_bundles
        # num_bundles : int
        #     The number of bundles that have been created so far.
        self.num_bundles = 0
        # nucleation_threshold : float
        #     Threshold above which nucleation energy results in nucleation.
        self.nucleation_threshold = 50.
        # agglomeration_threshold
        #     Threshold above which agglomeration energy results
        #     in agglomeration.
        self.agglomeration_threshold = self.nucleation_threshold
        # activity_threshold : float
        #     Threshold below which input activity is teated as zero.
        #     By ignoring the small activity values,
        #     computation gets much faster.
        self.activity_threshold = .1
        # bundles_full : bool
        #     If True, all the bundles in the Ziptie are full
        #     and learning stops. This is another way to speed up
        #     computation.
        self.bundles_full = False
        # cable_activities : array of floats
        #     The current set of input actvities.
        self.cable_activities = np.zeros(self.max_num_cables)
        # bundle_activities : array of floats
        #     The current set of bundle activities.
        self.bundle_activities = np.zeros(self.max_num_bundles)
        # nonbundle_activities : array of floats
        #     The set of input activities that do not contribute
        #     to any of the current bundle activities.
        self.nonbundle_activities = np.zeros(self.max_num_cables)

        # bundle_map_size : int
        #     The maximum number of non-zero entries in the bundle map.
        self.bundle_map_size = 8
        # bundle_map_cols, bundle_map_rows : array of ints
        #     To represent the sparse 2D bundle map, a pair of row and col
        #     arrays are used. Rows are bundle indices, and cols are
        #     feature indices.  The bundle map shows which cables
        #     are zipped together to form which bundles.
        self.bundle_map_rows = -np.ones(self.bundle_map_size).astype(int)
        self.bundle_map_cols = -np.ones(self.bundle_map_size).astype(int)
        # n_map_entries: int
        #     The total number of bundle map entries that
        #     have been created so far.
        self.n_map_entries = 0
        # agglomeration_energy : 2D array of floats
        #     The accumulated agglomeration energy for each bundle-cable pair.
        self.agglomeration_energy = np.zeros((self.max_num_bundles,
                                              self.max_num_cables))
        # nucleation_energy : 2D array of floats
        #     The accumualted nucleation energy associated
        #     with each cable-cable pair.
        self.nucleation_energy = np.zeros((self.max_num_cables,
                                           self.max_num_cables))


    def featurize(self, new_cable_activities, bundle_weights=None):
        """
        Calculate how much the cables' activities contribute to each bundle.

        Find bundle activities by taking the minimum input value
        in the set of cables in the bundle. The bulk of the computation
        occurs in ziptie_numba.find_bundle_activities.
        """
        self.cable_activities = new_cable_activities.copy()
        #self.nonbundle_activities = self.cable_activities.copy()
        #self.bundle_activities = np.zeros(self.max_num_bundles)
        self.bundle_activities = 1e3 * np.ones(self.max_num_bundles)
        if bundle_weights is None:
            bundle_weights = np.ones(self.max_num_bundles)
        if self.n_map_entries > 0:
            #nb.find_bundle_activities(
            #    self.bundle_map_rows[:self.n_map_entries],
            #    self.bundle_map_cols[:self.n_map_entries],
            #    self.nonbundle_activities,
            #    self.bundle_activities,
            #    bundle_weights, self.activity_threshold)
            for i_map_entry in xrange(self.n_map_entries):
                i_bundle = self.bundle_map_rows[i_map_entry]
                i_cable = self.bundle_map_cols[i_map_entry]
                self.bundle_activities[i_bundle] = (
                    np.minimum(self.bundle_activities[i_bundle],
                               self.cable_activities[i_cable]))
        self.bundle_activities[np.where(self.bundle_activities == 1e3)] = 0.
        self.bundle_activities *= bundle_weights
        # The residual cable_activities after calculating
        # bundle_activities are the nonbundle_activities.
        # Sparsify them by setting all the small values to zero.
        #self.nonbundle_activities[np.where(self.nonbundle_activities <
        #                                   self.activity_threshold)] = 0.
        return self.nonbundle_activities, self.bundle_activities


    def learn(self, cable_activities):
        """
        Update co-activity estimates and calculate bundle activity

        This step combines the projection of cables activities
        to bundle activities together with using the cable activities
        to incrementally train the Ziptie.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        if not self.bundles_full:
            self._create_new_bundles(cable_activities)
        if not self.bundles_full:
            self._grow_bundles(cable_activities)
        return


    def _create_new_bundles(self, cable_activities):
        """
        If the right conditions have been reached, create a new bundle.
        """
        # Incrementally accumulate nucleation energy.
        #nb.nucleation_energy_gather(self.nonbundle_activities,
        #                            self.nucleation_energy)
        nb.nucleation_energy_gather(cable_activities,
                                    self.nucleation_energy)

        # Don't accumulate nucleation energy between a cable and itself
        ind = np.arange(self.cable_activities.size).astype(int)
        self.nucleation_energy[ind, ind] = 0.

        # Don't accumulate nucleation energy between cables already
        # in the same bundle
        for i in range(self.n_map_entries):
            i_bundle = self.bundle_map_rows[i]
            i_cable = self.bundle_map_cols[i]
            j = 1
            j_bundle = self.bundle_map_rows[i + j]
            j_cable = self.bundle_map_cols[i + j]
            while j_bundle == i_bundle:
                self.nucleation_energy[i_cable, j_cable] = 0.
                self.nucleation_energy[j_cable, i_cable] = 0.
                j += 1
                j_bundle = self.bundle_map_rows[i + j]
                j_cable = self.bundle_map_cols[i + j]

        results = -np.ones(3)
        nb.max_dense(self.nucleation_energy, results)
        max_energy = results[0]
        cable_index_a = int(results[1])
        cable_index_b = int(results[2])

        # Add a new bundle if appropriate
        if max_energy > self.nucleation_threshold:
            self.bundle_map_rows[self.n_map_entries] = self.num_bundles
            self.bundle_map_cols[self.n_map_entries] = cable_index_a
            self.increment_n_map_entries()
            self.bundle_map_rows[self.n_map_entries] = self.num_bundles
            self.bundle_map_cols[self.n_map_entries] = cable_index_b
            self.increment_n_map_entries()
            self.num_bundles += 1

            print(' '.join(['    ', self.name,
                            'bundle', str(self.num_bundles),
                            'added with cables', str(cable_index_a),
                            str(cable_index_b)]))

            # Check whether the Ziptie's capacity has been reached.
            if self.num_bundles == self.max_num_bundles:
                self.bundles_full = True

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[cable_index_a, :] = 0.
            self.nucleation_energy[cable_index_b, :] = 0.
            self.nucleation_energy[:, cable_index_a] = 0.
            self.nucleation_energy[:, cable_index_b] = 0.
            self.agglomeration_energy[:, cable_index_a] = 0.
            self.agglomeration_energy[:, cable_index_b] = 0.


    def _grow_bundles(self, cable_activities):
        """
        Update an estimate of co-activity between all cables.
        """
        # Incrementally accumulate agglomeration energy.
        #nb.agglomeration_energy_gather(self.bundle_activities,
        #                               self.nonbundle_activities,
        #                               self.num_bundles,
        #                               self.agglomeration_energy)
        nb.agglomeration_energy_gather(self.bundle_activities,
                                       cable_activities,
                                       self.num_bundles,
                                       self.agglomeration_energy)

        # Don't accumulate agglomeration energy between cables already
        # in the same bundle
        val = 0.
        if self.n_map_entries > 0:
            nb.set_dense_val(self.agglomeration_energy,
                             self.bundle_map_rows[:self.n_map_entries],
                             self.bundle_map_cols[:self.n_map_entries],
                             val)

        results = -np.ones(3)
        nb.max_dense(self.agglomeration_energy, results)
        max_energy = results[0]
        cable_index = int(results[2])
        bundle_index = int(results[1])

        # Add a new bundle if appropriate
        if max_energy > self.agglomeration_threshold:
            # Find which cables are in the new bundle.
            cables = [cable_index]
            for i in range(self.n_map_entries):
                if self.bundle_map_rows[i] == bundle_index:
                    cables.append(self.bundle_map_cols[i])
            # Check whether the agglomeration is already in the bundle map.
            candidate_bundles = np.arange(self.num_bundles)
            for cable in cables:
                matches = np.where(self.bundle_map_cols == cable)[0]
                candidate_bundles = np.intersect1d(
                    candidate_bundles,
                    self.bundle_map_rows[matches],
                    assume_unique=True)
            if candidate_bundles.size != 0:
                # The agglomeration has already been used to create a
                # bundle. Ignore and reset they count. This can happen
                # under normal circumstances, because of how nonbundle
                # activities are calculated.
                self.agglomeration_energy[bundle_index, cable_index] = 0.
                return

            # Make a copy of the growing bundle.
            for i in range(self.n_map_entries):
                if self.bundle_map_rows[i] == bundle_index:
                    self.bundle_map_rows[self.n_map_entries] = self.num_bundles
                    self.bundle_map_cols[self.n_map_entries] = (
                        self.bundle_map_cols[i])
                    self.increment_n_map_entries()
            # Add in the new cable.
            self.bundle_map_rows[self.n_map_entries] = self.num_bundles
            self.bundle_map_cols[self.n_map_entries] = cable_index
            self.increment_n_map_entries()
            self.num_bundles += 1

            if self.debug:
                print(' '.join(['    ', self.name,
                                'bundle', str(self.num_bundles),
                                'added: bundle', str(bundle_index),
                                'and cable', str(cable_index)]))

            # Check whether the Ziptie's capacity has been reached.
            if self.num_bundles == self.max_num_bundles:
                self.bundles_full = True

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[cable_index, :] = 0.
            self.nucleation_energy[cable_index, :] = 0.
            self.nucleation_energy[:, cable_index] = 0.
            self.nucleation_energy[:, cable_index] = 0.
            self.agglomeration_energy[:, cable_index] = 0.
            self.agglomeration_energy[bundle_index, :] = 0.


    def increment_n_map_entries(self):
        """
        Add one to n_map entries and grow the bundle map as needed.
        """
        self.n_map_entries += 1
        if self.n_map_entries >= self.bundle_map_size:
            self.bundle_map_size *= 2
            self.bundle_map_rows = tools.pad(self.bundle_map_rows,
                                             self.bundle_map_size,
                                             val=-1, dtype='int')
            self.bundle_map_cols = tools.pad(self.bundle_map_cols,
                                             self.bundle_map_size,
                                             val=-1, dtype='int')


    def get_index_projection(self, bundle_index):
        """
        Project bundle_index down to its cable indices.

        Parameters
        ----------
        bundle_index : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection : array of floats
            An array of zeros and ones, representing all the cables that
            contribute to the bundle. The values projection
            corresponding to all the cables that contribute are 1.
        """
        projection = np.zeros(self.max_num_cables)
        for i in range(self.n_map_entries):
            if self.bundle_map_rows[i] == bundle_index:
                projection[self.bundle_map_cols[i]] = 1.
        return projection


    def get_index_projection_cables(self, bundle_index):
        """
        Project bundle_index down to its cable indices.

        Parameters
        ----------
        bundle_index : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection_indices : array of ints
            An array of cable indices, representing all the cables that
            contribute to the bundle.
        """
        projection = []
        for i in range(self.n_map_entries):
            if self.bundle_map_rows[i] == bundle_index:
                projection.append(self.bundle_map_cols[i])
        projection_indices = np.array(projection)
        return projection_indices


    def project_bundle_activities(self, bundle_activities):
        """
        Take a set of bundle activities and project them to cable activities.
        """
        cable_activities = np.zeros(self.max_num_cables)
        for i in range(self.n_map_entries):
            i_bundle = self.bundle_map_rows[i]
            i_cable = self.bundle_map_cols[i]
            cable_activities[i_cable] = max(cable_activities[i_cable],
                                            bundle_activities[i_bundle])
        return cable_activities


    def visualize(self):
        """
        Turn the state of the Ziptie into an image.
        """
        print(self.name)
        # First list the bundles and the cables in each.
        i_bundles = self.bundle_map_rows[:self.n_map_entries]
        i_cables = self.bundle_map_cols[:self.n_map_entries]
        i_bundles_unique = np.unique(i_bundles)
        if i_bundles_unique is not None:
            for i_bundle in i_bundles_unique:
                b_cables = list(np.sort(i_cables[np.where(
                    i_bundles == i_bundle)[0]]))
                print(' '.join(['    bundle', str(i_bundle),
                                'cables:', str(b_cables)]))

        plot = False
        if plot:
            if self.n_map_entries > 0:
                # Render the bundle map.
                bundle_map = np.zeros((self.max_num_cables,
                                       self.max_num_bundles))
                nb.set_dense_val(bundle_map,
                                 self.bundle_map_rows[:self.n_map_entries],
                                 self.bundle_map_cols[:self.n_map_entries], 1.)
                tools.visualize_array(bundle_map,
                                      label=self.name + '_bundle_map')

                # Render the agglomeration energy.
                label = '_'.join([self.name, 'agg_energy'])
                tools.visualize_array(self.agglomeration_energy, label=label)
                plt.xlabel(str(np.max(self.agglomeration_energy)))

                # Render the nucleation energy.
                label = '_'.join([self.name, 'nuc_energy'])
                tools.visualize_array(self.nucleation_energy, label=label)
                plt.xlabel(str(np.max(self.nucleation_energy)))
