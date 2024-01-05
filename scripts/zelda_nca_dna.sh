#!/bin/bash

# nca + dna sampler
# CUDA_VISIBLE_DEVICES="1" python -m bin.train \
#   experiment=qd_dna_zelda_gen \
#   model=nca_dna_all_alive \
#   +model@model.dna=dna_iid_sampler

# # nca + dna list
# CUDA_VISIBLE_DEVICES="2" python -m bin.train experiment=qd_dna_zelda_gen \
#   task.popsize=25 task.qd_algorithm.emitter.batch_size=25 \
#   +model@model.dna=dna_list model.dna.n_dnas=\${task.popsize}

# nca + dna sampler (using CMA emitter in the inner loop)
CUDA_VISIBLE_DEVICES="0" python -m bin.train \
  experiment=qd_dna_zelda_gen \
  +model@model.dna=dna_iid_sampler \
  qd@task.qd_algorithm=cmame  \
  +task.qd_algorithm.emitter.genotype_dim=32

# # dummy strategy
# CUDA_VISIBLE_DEVICES="3" python -m bin.train experiment=dummy_qd_dna_zelda_gen \
#   qd@task.qd_algorithm=cmame \
#   +task.qd_algorithm.emitter.genotype_dim=32 \
#   +model@model.dna=dna_iid_sampler
#
# CUDA_VISIBLE_DEVICES="1" python -m bin.train \
#   experiment=qd_dna_zelda_gen \
#   qd@task.qd_algorithm=cmame \
#   +qd/score_aggregator@task.score_aggregator=coverage_only \
#   +task.qd_algorithm.emitter.genotype_dim=32 \
#   +model@model.dna=dna_iid_sampler

# maximize pairwise correspondance between genotypes and phontypes
CUDA_VISIBLE_DEVICES="1" python -m bin.train \
  experiment=qd_dna_zelda_gen \
  qd@task.qd_algorithm=cmame \
  +qd/score_aggregator@task.score_aggregator=score_plus_pairwise_consistency \
  +task.qd_algorithm.emitter.genotype_dim=32 \
  +model@model.dna=dna_iid_sampler
