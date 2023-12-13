#!/bin/bash

# nca + dna sampler
CUDA_VISIBLE_DEVICES="1" python -m bin.train experiment=qd_dna_zelda_gen \
  +model@model.dna=dna_iid_sampler

# nca + dna list
CUDA_VISIBLE_DEVICES="2" python -m bin.train experiment=qd_dna_zelda_gen \
  task.popsize=25 task.qd_algorithm.emitter.batch_size=25 \
  +model@model.dna=dna_list model.dna.n_dnas=\${task.popsize}
