from ui.window import Window
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, help='Select the type of activate function')
    parser.add_argument('--alpha', type=float, default="0.01", help='Input the alpha for ReLU / Leaky ReLU active function')
    args = parser.parse_args()

    game_window = Window(args.function, args.alpha)

