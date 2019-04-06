from calc.Calc import Calc


def main():
    calc = Calc(3, 7)

    print("{} + {} = {}".format(calc.first, calc.second, calc.sum()))
    print("{} + {} = {}".format(calc.first, calc.second, calc.mul()))
    print("{} + {} = {}".format(calc.first, calc.second, calc.minus()))
    print("{} + {} = {}".format(calc.first, calc.second, calc.div()))


if __name__ == '__main__':
    main()