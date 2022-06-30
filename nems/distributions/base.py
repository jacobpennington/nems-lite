import numpy as np


class Distribution:
    """Base class for a Distribution."""
    
    @classmethod
    def value_to_string(cls, value):
        if value.ndim == 0:
            return 'scalar'
        else:
            shape = ', '.join(str(v) for v in value.shape)
            return 'array({})'.format(shape)

    def mean(self):
        """Return the expected value of the distribution."""
        return self.distribution.mean()

    def percentile(self, percentile):
        """Calculate the percentile.

        Parameters
        ----------
        percentile : float [0, 1]
            Probability at which the result is calculated. Should be specified as
            a fraction in the range 0 ... 1 rather than a percent.

        Returns
        -------
        value : float
            Value of random variable at given percentile

        Examples
        --------
        For some distributions (e.g., Normal), the bounds will be +/- infinity.
        In those situations, you can request that you get the bounds for the 99%
        interval to get a slightly more reasonable constraint that can be passed
        to the fitter.
        >>> from nems.distributions.api import Normal
        >>> prior = Normal(mu=0, sd=1)
        >>> lower = prior.percentile(0.005)
        >>> upper = prior.percentile(0.995)

        """
        return self.distribution.ppf(percentile)

    @property
    def shape(self):
        return self.mean().shape

    def sample(self, n=None, bounds=None):
        if n is None:
            n = 1
        size = [n] + list(self.shape)
        good_sample = np.full(shape=size, fill_value=np.nan)

        while np.sum(np.isnan(good_sample)) > 0:
            sample = self.distribution.rvs(size=size)
            if bounds is not None:
                lower, upper = bounds
                keep = (sample >= lower) | (sample <= upper)
                good_sample[keep] = sample[keep]
            else:
                good_sample = sample
                break

        # Drop first dimension if n = 1
        final_sample = np.squeeze(good_sample, axis=0)

        return final_sample

    def tolist(self):
        d = self.__dict__
        if 'distribution' in d:
            del d['distribution']
        name = type(self).__name__
        l = [name, d]
        return l
