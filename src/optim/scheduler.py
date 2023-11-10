import optax


def cosine_schedule_with_restarts(
      transition_steps: int,
      peak_value: float,
      pct_start: float,
      div_factor: float,
      final_div_factor: float,
      n_restarts: int,
      peak_decay: float = 1.0
  ):
    schedulers = [
        optax.cosine_onecycle_schedule(
            transition_steps,
            peak_value * peak_decay ** i,
            pct_start,
            div_factor,
            final_div_factor)
        for i in range(n_restarts)
    ]

    boundaries = [transition_steps * (i + 1) for i in range(n_restarts - 1)]

    schedule = optax.join_schedules(schedulers, boundaries)
    return schedule
