#!/bin/bash

python -m bin.train experiment=qd_dna_zelda_gen \
  strategy@trainer.strategy=dummy trainer.strategy.popsize=2 \
  task.n_iters=100
