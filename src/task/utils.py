import equinox as eqx


class InputSelect(eqx.Module):
    idx: int = 0
    def __call__(self, inputs, targets):
        return inputs[self.idx], targets
