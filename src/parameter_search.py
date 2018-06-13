from train import main
from config import

hidden_state_size = [512, 1024]
attention = [True, False]
share_weights = [True, False]
dropout = [True, False]
normalize = [True, False]


for h in hidden_state_size:
    for a in attention:
        for s in share_weights:
            for n in normalize:
                pass


