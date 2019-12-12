'''
Features used by AlphaGo, in approximate order of importance.
Feature                 # Notes
Stone colour            3 Player stones; oppo. stones; empty  
Ones                    1 Constant plane of 1s 
    (Because of convolution w/ zero-padding, this is the only way the NN can know where the edge of the board is!!!)
Turns since last move   8 How many turns since a move played
Liberties               8 Number of liberties
Capture size            8 How many opponent stones would be captured
Self-atari size         8 How many own stones would be captured
Liberties after move    8 Number of liberties after this move played
ladder capture          1 Whether a move is a successful ladder cap
Ladder escape           1 Whether a move is a successful ladder escape
Sensibleness            1 Whether a move is legal + doesn't fill own eye
Zeros                   1 Constant plane of 0s

All features with 8 planes are 1-hot encoded, with plane i marked with 1 
only if the feature was equal to i. Any features >= 8 would be marked as 8.
'''

import numpy as np
import go
from utils import product

# Resolution/truncation limit for one-hot features
P = 8

def make_onehot(feature, planes):
    onehot_features = np.zeros(feature.shape + (planes,), dtype=np.uint8)
    capped = np.minimum(feature, planes)
    onehot_index_offsets = np.arange(0, product(onehot_features.shape), planes) + capped.ravel()
    # A 0 is encoded as [0,0,0,0], not [1,0,0,0], so we'll
    # filter out any offsets that are a multiple of $planes
    # A 1 is encoded as [1,0,0,0], not [0,1,0,0], so subtract 1 from offsets
    nonzero_elements = (capped != 0).ravel()
    nonzero_index_offsets = onehot_index_offsets[nonzero_elements] - 1
    onehot_features.ravel()[nonzero_index_offsets] = 1
    return onehot_features

def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f
    return deco

@planes(3)
def stone_color_feature(position):
    board = position.board
    features = np.zeros([go.N, go.N, 3], dtype=np.uint8)
    if position.to_play == go.BLACK:
        features[board == go.BLACK, 0] = 1
        features[board == go.WHITE, 1] = 1
    else:
        features[board == go.WHITE, 0] = 1
        features[board == go.BLACK, 1] = 1

    features[board == go.EMPTY, 2] = 1
    return features

@planes(1)
def ones_feature(position):
    return np.ones([go.N, go.N, 1], dtype=np.uint8)

@planes(P)
def recent_move_feature(position):
    onehot_features = np.zeros([go.N, go.N, P], dtype=np.uint8)
    for i, player_move in enumerate(reversed(position.recent[-P:])):
        _, move = player_move # unpack the info from position.recent
        if move is not None:
            onehot_features[move[0], move[1], i] = 1
    return onehot_features

@planes(P)
def liberty_feature(position):
    return make_onehot(position.get_liberties(), P)

@planes(P)
def would_capture_feature(position):
    features = np.zeros([go.N, go.N], dtype=np.uint8)
    for g in position.lib_tracker.groups.values():
        if g.color == position.to_play:
            continue
        if len(g.liberties) == 1:
            last_lib = list(g.liberties)[0]
            # += because the same spot may capture more than 1 group.
            features[last_lib] += len(g.stones)
    return make_onehot(features, P)

DEFAULT_FEATURES = [
    stone_color_feature,
    ones_feature,
    liberty_feature,
    recent_move_feature,
    would_capture_feature,
]

def extract_features(position, features=DEFAULT_FEATURES):
    return np.concatenate([feature(position) for feature in features], axis=2)

def bulk_extract_features(positions, features=DEFAULT_FEATURES):
    num_positions = len(positions)
    num_planes = sum(f.planes for f in features)
    output = np.zeros([num_positions, go.N, go.N, num_planes], dtype=np.uint8)
    for i, pos in enumerate(positions):
        output[i] = extract_features(pos, features=features)
    return output
