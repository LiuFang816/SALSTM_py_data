"""
.. _tut_compute_inverse_erf:

========================================
Compute inverse solution for evoked data
========================================

Here we'll use our knowledge from the other examples and tutorials
to compute an inverse solution and apply it on event related fields.
"""
# Author: Denis A. Enegemann
# License: BSD 3 clause

import os.path as op
import mne
import hcp
from hcp import preprocessing as preproc

##############################################################################
# we assume our data is inside a designated folder under $HOME
storage_dir = op.expanduser('~/mne-hcp-data')
hcp_path = op.join(storage_dir, 'HCP')
recordings_path = op.join(storage_dir, 'hcp-meg')
subjects_dir = op.join(storage_dir, 'hcp-subjects')
subject = '105923'  # our test subject
data_type = 'task_working_memory'
run_index = 0

##############################################################################
# We're reading the evoked data.
# These are the same as in :ref:`tut_plot_evoked`

hcp_evokeds = hcp.read_evokeds(onset='stim', subject=subject,
                                   data_type=data_type, hcp_path=hcp_path)
for evoked in hcp_evokeds:
    if not evoked.comment == 'Wrkmem_LM-TIM-face_BT-diff_MODE-mag':
        continue

##############################################################################
# We'll now use a convenience function to get our forward and source models
# instead of computing them by hand.

src_outputs = hcp.anatomy.compute_forward_stack(
    subject=subject, subjects_dir=subjects_dir,
    hcp_path=hcp_path, recordings_path=recordings_path,
    # speed up computations here. Setting `add_dist` to True may improve the
    # accuracy.
    src_params=dict(add_dist=False),
    info_from=dict(data_type=data_type, run_index=run_index))

fwd = src_outputs['fwd']

##############################################################################
# Now we can compute the noise covariance. For this purpose we will apply
# the same filtering as was used for the computations of the ERF in the first
# place. See also :ref:`tut_reproduce_erf`.

raw_noise = hcp.read_raw(subject=subject, hcp_path=hcp_path,
                         data_type='noise_empty_room')
raw_noise.load_data()

# apply ref channel correction and drop ref channels
preproc.apply_ref_correction(raw_noise)

# Note: MNE complains on Python 2.7
raw_noise.filter(0.50, None, method='iir',
                 iir_params=dict(order=4, ftype='butter'), n_jobs=1)
raw_noise.filter(None, 60, method='iir',
                 iir_params=dict(order=4, ftype='butter'), n_jobs=1)

##############################################################################
# Note that using the empty room noise covariance will inflate the SNR of the
# evkoked and renders comparisons  to `baseline` rather uninformative.
noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')


##############################################################################
# Now we assemble the inverse operator, project the data and show the results
# on the `fsaverage` surface, the freesurfer average brain.

inv_op = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd, noise_cov=noise_cov)

stc = mne.minimum_norm.apply_inverse(  # these data have a pretty high SNR and
    evoked, inv_op, method='MNE', lambda2=1./9.**2)  # 9 is a lovely number.

stc = stc.to_original_src(
    src_outputs['src_fsaverage'], subjects_dir=subjects_dir)

brain = stc.plot(subject='fsaverage', subjects_dir=subjects_dir, hemi='both')
brain.set_time(145)  # we take the peak seen in :ref:`tut_plot_evoked` and
brain.show_view('caudal')  # admire wide spread visual activation.
