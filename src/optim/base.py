import optax


def init_optimizer(optim_fn, inject_hyperparams=True, *args, **kwargs):
    if inject_hyperparams:
        return optax.inject_hyperparams(optim_fn)(*args, **kwargs)
    return optim_fn(*args, **kwargs)
