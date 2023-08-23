from envs import TidyUpDish, DrillScrew


def main():
    env = TidyUpDish()

    while not env.viewer.closed:
        action = None
        env.step(action)


if __name__ == '__main__':
    main()