from nems import ModelSpec

class LN_STRF(ModelSpec):
    '''
    A ModelSpec with the following modules:
        1) WeightChannels(shape=(4, n_channels), parameterization='gaussian')
        2) FIR(shape=(4, 15), parameterization='P3Z1')
        3) LevelShift(),
        4) DoubleExponential()

    Based on the best-performing model from
    Thorson, Lienard and David (2015)
    doi: 10.1371/journal.pcbi.1004628

    '''

    def __init__(n_channels):
        # Need to know number of spectral channels in the stimulus
        self.n_channels = n_channels

    # TODO: everything else, this is just to illustrate the idea.
