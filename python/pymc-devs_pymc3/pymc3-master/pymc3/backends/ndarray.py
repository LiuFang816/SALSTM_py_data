"""NumPy array trace backend

Store sampling values in memory as a NumPy array.
"""
import numpy as np
from ..backends import base


class NDArray(base.BaseTrace):
    """NDArray trace object

    Parameters
    ----------
    name : str
        Name of backend. This has no meaning for the NDArray backend.
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """

    supports_sampler_stats = True

    def __init__(self, name=None, model=None, vars=None):
        super(NDArray, self).__init__(name, model, vars)
        self.draw_idx = 0
        self.draws = None
        self.samples = {}
        self._stats = None

    # Sampling methods

    def setup(self, draws, chain, sampler_vars=None):
        """Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        sampler_vars : list of dicts
            Names and dtypes of the variables that are
            exported by the samplers.
        """
        super(NDArray, self).setup(draws, chain, sampler_vars)

        self.chain = chain
        if self.samples:  # Concatenate new array if chain is already present.
            old_draws = len(self)
            self.draws = old_draws + draws
            self.draws_idx = old_draws
            for varname, shape in self.var_shapes.items():
                old_var_samples = self.samples[varname]
                new_var_samples = np.zeros((draws, ) + shape,
                                           self.var_dtypes[varname])
                self.samples[varname] = np.concatenate((old_var_samples,
                                                        new_var_samples),
                                                       axis=0)
        else:  # Otherwise, make array of zeros for each variable.
            self.draws = draws
            for varname, shape in self.var_shapes.items():
                self.samples[varname] = np.zeros((draws, ) + shape,
                                                 dtype=self.var_dtypes[varname])

        if sampler_vars is None:
            return

        if self._stats is None:
            self._stats = []
            for sampler in sampler_vars:
                data = dict()
                self._stats.append(data)
                for varname, dtype in sampler.items():
                    data[varname] = np.zeros(draws, dtype=dtype)
        else:
            for data, vars in zip(self._stats, sampler_vars):
                if vars.keys() != data.keys():
                    raise ValueError("Sampler vars can't change")
                old_draws = len(self)
                for varname, dtype in vars.items():
                    old = data[varname]
                    new = np.zeros(draws, dtype=dtype)
                    data[varname] = np.concatenate([old, new])

    def record(self, point, sampler_stats=None):
        """Record results of a sampling iteration.

        Parameters
        ----------
        point : dict
            Values mapped to variable names
        """
        for varname, value in zip(self.varnames, self.fn(point)):
            self.samples[varname][self.draw_idx] = value

        if self._stats is not None and sampler_stats is None:
            raise ValueError("Expected sampler_stats")
        if self._stats is None and sampler_stats is not None:
            raise ValueError("Unknown sampler_stats")
        if sampler_stats is not None:
            for data, vars in zip(self._stats, sampler_stats):
                for key, val in vars.items():
                    data[key][self.draw_idx] = val
        self.draw_idx += 1

    def _get_sampler_stats(self, varname, sampler_idx, burn, thin):
        return self._stats[sampler_idx][varname][burn::thin]

    def close(self):
        if self.draw_idx == self.draws:
            return
        # Remove trailing zeros if interrupted before completed all
        # draws.
        self.samples = {var: vtrace[:self.draw_idx]
                        for var, vtrace in self.samples.items()}
        if self._stats is not None:
            self._stats = [{var: trace[:self.draw_idx] for var, trace in stats.items()}
                           for stats in self._stats]

    # Selection methods

    def __len__(self):
        if not self.samples:  # `setup` has not been called.
            return 0
        return self.draw_idx

    def get_values(self, varname, burn=0, thin=1):
        """Get values from trace.

        Parameters
        ----------
        varname : str
        burn : int
        thin : int

        Returns
        -------
        A NumPy array
        """
        return self.samples[varname][burn::thin]

    def _slice(self, idx):
        # Slicing directly instead of using _slice_as_ndarray to
        # support stop value in slice (which is needed by
        # iter_sample).

        # Only the first `draw_idx` value are valid because of preallocation
        idx = slice(*idx.indices(len(self)))

        sliced = NDArray(model=self.model, vars=self.vars)
        sliced.chain = self.chain
        sliced.samples = {varname: values[idx]
                          for varname, values in self.samples.items()}
        sliced.sampler_vars = self.sampler_vars
        if self._stats is None:
            return sliced
        sliced._stats = []
        for vars in self._stats:
            var_sliced = {}
            sliced._stats.append(var_sliced)
            for key, vals in vars.items():
                var_sliced[key] = vals[idx]

        sliced.draw_idx = idx.stop - idx.start
        return sliced

    def point(self, idx):
        """Return dictionary of point values at `idx` for current chain
        with variable names as keys.
        """
        idx = int(idx)
        return {varname: values[idx]
                for varname, values in self.samples.items()}


def _slice_as_ndarray(strace, idx):
    if idx.start is None:
        burn = 0
    else:
        burn = idx.start
    if idx.step is None:
        thin = 1
    else:
        thin = idx.step

    sliced = NDArray(model=strace.model, vars=strace.vars)
    sliced.chain = strace.chain
    sliced.samples = {v: strace.get_values(v, burn=burn, thin=thin)
                      for v in strace.varnames}
    return sliced
