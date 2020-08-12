from miniature_gbt import MiniatureGbtModel

import argparse

parser = argparse.ArgumentParser(description='Process params')
parser.add_argument('--line', dest='line', type=int, default=2,
                    help='number of line')
parser.add_argument('--checkpoint', dest='checkpoint', type=str, default="models/v0/gen_peotry/cp.ckpt",
                    help='checkpoint model path')


def speak(args):
    model = MiniatureGbtModel(
            batch_size=32,
            epochs=10,
            vocab_size=20000,
            maxlen=80,
            num_tokens_generated=40,
            n_dim=256,
            num_heads=2,
            feed_forward_dim=256,
            checkpoint_path=args.checkpoint,
    )
    for i in range(args.line):
        print('Sphinx:')
        text_gen_call = model.make_text_gen_callback('the world')
        text = text_gen_call.gen_text()
        print(text)


def main():
    args = parser.parse_args()
    speak(args)


if __name__ == '__main__':
    main()
