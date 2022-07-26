from tkinter import FALSE
import numpy as np

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter


class StaticNonlinearity(Layer):

    def __init__(self, shape, **kwargs):
        """Apply a nonlinear transformation to input(s).
        
        TODO: Test if current implementations will work with higher dim data.

        Parameters
        ----------
        shape : N-tuple (usually N=1)
            Determines the shape of each Parameter in `.parameters`.
            First dimension should match the spectral dimenion of the input.
            Note that higher-dimesional shapes are also allowed, but overall
            Layer design is intended for 1-dimensional shapes.

        Attributes
        ----------
        skip_nonlinearity : bool
            If True, don't apply `nonlinearity` during `evaluate`. Still apply
            `input += shift`, if `'shift' in StaticNonlinearity.parameters`.
        
        """
        self.shape = shape
        super().__init__(**kwargs)
        self.skip_nonlinearity = False

    def evaluate(self, *inputs):
        """Apply `nonlinearity` to input(s). This should not be overwriten."""
        if not self.skip_nonlinearity:
            output = self.nonlinearity(inputs)
        else:
            # TODO: This works for time on 0-axis and 1-dim parameters,
            #       but need to add option to make this more general.
            # If there's a `shift` parameter for the subclassed nonlinearity,
            # still apply that. Otherwise, pass through inputs.
            output = [inputs + self.parameters.get('shift', 0).values]
        return output

    def nonlinearity(self, *inputs):
        """Pass through input(s). Subclasses should overwrite this."""
        return inputs


class LevelShift(StaticNonlinearity):
    """Applies a scalar shift to each input channels.
    
    Notes
    -----
    While `LevelShift.evaluate` is linear, this Layer is grouped with
    `StaticNonlinearity` because of its close relation to these other layers.
    In short, we have found in the past that it is often helpful to "turn off"
    a nonlinearity during fitting, but still shift the input.
    
    """

    def initial_parameters(self):
        """Get initial values for `StaticNonlinearity.parameters`.
        
        Layer parameters
        ----------------
        shift : scalar or ndarray
            Value(s) that are added to input(s) prior to rectification. Shape
            (N,) must match N channels per input.
            Prior:  TODO
        
        """
        # TODO: explain choice of priors.
        prior = Normal(
            np.zeros(shape=self.shape), 
            np.ones(shape=self.shape)/100
            )

        shift = Parameter('shift', shape=self.shape, prior=prior)
        return Phi(shift)

    def nonlinearity(self, *inputs):
        """constant shift

        Notes
        -----
        Simply add a constant shift to the signal

        """
        shift, = self.get_parameter_values()
        return [x + shift for x in inputs]

    @layer('lvl')
    def from_keyword(keyword):
        """Construct LevelShift from a keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.

        Returns
        -------
        LevelShift

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return LevelShift(shape=shape)


class DoubleExponential(StaticNonlinearity):

    def initial_parameters(self):
        """Get initial values for `DoubleExponential.parameters`.
        
        Layer parameters
        ----------------
        base : scalar or ndarray
            Y-axis height of the center of the sigmoid.
            Shape (N,) must match N input channels (same for other parameters),
            such that one sigmoid transformation is applied to each channel.
            Prior:  Normal(mean=0, sd=1)
            Bounds: TODO
        amplitude : scalar or ndarray
            Y-axis distance from ymax asymptote to ymin asymptote
            Prior:  Normal(mean=5, sd=1.5)
            Bounds: TODO
        shift : scalar or ndarray
            Centerpoint of the sigmoid along x axis
            Prior:  Normal(mean=0, sd=1)
            Bounds: TODO
        kappa : scalar or ndarray
            Sigmoid curvature. Larger numbers mean steeper slop.
            Prior:  Normal(mean=1, sd=10)
            Bounds: TODO

        Returns
        -------
        nems.layers.base.Phi
        
        """
        # TODO: explain choices for priors.
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)
        phi = Phi(
            Parameter('base', shape=self.shape, prior=Normal(zero, one)),
            Parameter('amplitude', shape=self.shape,
                      prior=Normal(5*zero, 1.5*one)),
            Parameter('shift', shape=self.shape, prior=Normal(zero, one)),
            Parameter('kappa', shape=self.shape, prior=Normal(one, 10*one))
            )
        return phi

    def nonlinearity(self, *inputs):
        """Apply double exponential sigmoid to input x: $b+a*exp[-exp(k(x-s)]$.
        
        See Thorson, Liénard, David (2015).
        
        """
        base, amplitude, shift, kappa = self.get_parameter_values()
        output = []
        for x in inputs:
            y = base + amplitude * np.exp(-np.exp(  # double exponential
                    np.array(-np.exp(kappa)) * (x - shift)  # exp(kappa) > 0
                    ))
            # TODO: fix dim alignment so that the previous transformation
            #       doesn't add an extra axis (presumably something to do with
            #       array broadcasting).
            z = np.squeeze(y, axis=0)
            output.append(z)
        return output

    @layer('dexp')
    def from_keyword(keyword):
        """Construct DoubleExponential from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.

        Returns
        -------
        DoubleExponential

        See also
        --------
        Layer.from_keyword
        
        """
        shape = ()
        options = keyword.split('.')
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])
        
        return DoubleExponential(shape=shape)


class RectifiedLinear(StaticNonlinearity):

    def initial_parameters(self):
        """Get initial values for `RectifiedLinear.parameters`.
        
        Layer parameters
        ----------------
        shift : scalar or ndarray
            Value(s) that are added to input(s) prior to rectification. Shape
            (N,) must match N channels per input.
            Prior:  Normal(mean=-0.1, sd=1/sqrt(N))
        
        """
        # TODO: explain choice of prior.
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)
        prior = {'mean': zero-0.1, 'sd': one/np.sqrt(self.shape[0])}
        phi = Phi(Parameter('shift', shape=self.shape, prior=Normal(**prior)))

        return phi

    def nonlinearity(self, *inputs):
        """Set input values to 0 if <= `-1*shift`.
        
        Notes
        -----
        The negative of `shift` is used so that its interpretation in
        `StaticNonlinearity.evaluate` is the same as for other subclasses.
        
        """
        # TODO: add ability to multiply Parameter by scalar
        return [x * (x > -self.parameters['shift'].values) for x in inputs]

    @layer('relu')
    def from_keyword(keyword):
        """Construct RectifiedLinear from a keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.

        Returns
        -------
        RectifiedLinear

        See also
        --------
        Layer.from_keyword

        """
        options = keyword.split('.')
        no_shift = False
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])
            elif op == 'f':
                no_shift = True

        relu = RectifiedLinear(shape=shape)
        if no_shift:
            relu.set_permanent_values(shift=0)

        return relu

# Optional alias
ReLU = RectifiedLinear
