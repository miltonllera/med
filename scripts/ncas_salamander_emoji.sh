#!/bin/bash

# Testing evolutionary training of NCAs to generate a salamander emoji (as in Mordvintsev et al.)

# standard nca
python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200

# standard nca but fitted using LGA (Lange et al., 2023)
python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200 strategy@trainer.strategy=lga

# # squashing outputs
# python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200 model=nca_bounded_output

# # squashing updates
# python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200 model=nca_bounded_updates

# # squashing both
# python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200 model=nca_all_bounded
