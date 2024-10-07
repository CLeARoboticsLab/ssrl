def train(epochs,
          num_timesteps=2**15,
          episode_length=128,
          num_evals=3,
          seed=0):
    args = locals()
    initizalize_training(args)


def initizalize_training(args: dict):
    print(args['epochs'])


if __name__ == '__main__':
    train(10)
