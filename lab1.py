from argparse import ArgumentParser

import step4
import step5
import step6


def main():
    # Run the following line in terminal
    # python lab1.py –l MLP.8.pth

    # set up arg parser and model
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-l', '--saved_model')
    args = arg_parser.parse_args()

    my_model = args.saved_model
    my_model.eval()

    # run step 4
    step4.run(model=my_model)

    # run step 5
    step5.run(model=my_model)

    # run step 6
    step6.run(model=my_model)

    # end main
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
