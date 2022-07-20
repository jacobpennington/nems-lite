from .base import Layer

class StateGain(Layer):
    def __init__(self, **kwargs):
        self.state_name = 'state'  # See Layer.__init__

    def evaluate(self, arg1, arg2, state):
        # TODO
        pass
