#!/bin/bash

# standard nca
python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200

# squashing outputs
python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200 model=nca_bounded_output

# squashing updates
python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200 model=nca_bounded_updates

# squashing both
python -m bin.train experiment=single_emoji_nca_evo trainer.strategy.popsize=200 model=nca_all_bounded
