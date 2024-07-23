# Simple ingredient for offline reinforcement learning
This is pytorch implementation of algorithms from the paper

Simple ingredient for offline reinforcement learning

by Edoardo Cetin, Andrea Tirinzoni, Matteo Pirotta, Alessandro Lazaric, Yann Ollivier, Ahmed Touati

[Paper](https://arxiv.org/pdf/2403.13097)

# Requirements

* Create a new conda environment: `conda create --name offline_rl python=3.18`
* Activate newly created enviornment: `conda activate offline_rl`
* Install pytorch: `conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia`
* Sanity check: `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())`
* Install dependencies `pip install -r requirements.txt`

# License
The majority of offline_rl is licensed under CC-BY-NC, however portions of the project are available under separate license terms: DMC Control and D4RL are licensed under the Apache license.