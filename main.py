from envs.tidy_up_dish import TidyUpDish
from envs.drill_screw import DrillScrew


def main():
    env = TidyUpDish()

    while not env.viewer.closed:
        action = None
        env.step(action)


if __name__ == '__main__':
    main()