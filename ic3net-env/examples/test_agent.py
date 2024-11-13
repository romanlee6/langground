import numpy as np

def obs_to_state(obs, vision=1, dim=5):
    self_vector = obs[vision][vision]
    index = np.argmax(self_vector[0:dim * dim])
    predator_y = index / 5
    predator_x = index % 5
