from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, dest="seed")
    parser.add_argument("--gpus", nargs="+", default=0, dest="gpus")
    # add dataset name if dev or test subcommands
    parser.add_argument("--output_dir", default="results/model_name", type=str, dest="output_dir")
    parser.add_argument("--debug", action="store_true", dest="debug")

    subparsers = parser.add_subparsers(title="subcommands", description="valid subcommands", dest="subcommand")
    train_subparser = subparsers.add_parser(name="train")
    train_parser_builder(train_subparser)
    dev_subparser = subparsers.add_parser(name="dev")
    dev_parser_builder(dev_subparser)
    test_subparser = subparsers.add_parser(name="test")
    test_parser_builder(test_subparser)

    args = parser.parse_args()

    return args


def train_parser_builder(parser: ArgumentParser) -> None:
    # add train specific argument
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("dev_data_path", type=str)
    parser.add_argument("test_data_path", type=str)
    parser.add_argument("--mini_batch_size", default=4, type=int, dest="mini_batch_size")
    parser.add_argument("--accumulation_steps", default=4, type=int, dest="accumulation_steps")
    parser.add_argument("--evaluate_batch_size", default=8, type=int, dest="evaluate_batch_size")
    parser.add_argument("--lr", default=5e-5, type=float, dest="lr")
    parser.add_argument("--warmup_steps", default=400, type=int, dest="warmup_steps")
    parser.add_argument("--clip_grad_norm", default=400., type=float, dest="clip_grad_norm")
    parser.add_argument("--epochs", default=80, type=int, dest="epochs")
    parser.add_argument("--patience", default=6, type=int, dest="patience")
    parser.add_argument("--metric", default="+BLEU-2", type=str, dest="metric")

    add_model_subparsers(parser)


def dev_parser_builder(parser: ArgumentParser) -> None:
    # add dev specific argument
    parser.add_argument("model_path", type=str)
    parser.add_argument("dev_data_path", type=str)

    # add model subparsers
    add_model_subparsers(parser)


def test_parser_builder(parser: ArgumentParser) -> None:
    # add dev specific argument
    parser.add_argument("model_path", type=str)
    parser.add_argument("dev_data_path", type=str)

    # add model subparsers
    add_model_subparsers(parser)


def add_model_subparsers(parser: ArgumentParser) -> None:
    model_parsers = parser.add_subparsers(title="model names", description="valid model names", dest="model_name")

    pipeline_text_generater_parser = model_parsers.add_parser(name="pipeline_text_generater")
    pipeline_text_generater_parser_builder(pipeline_text_generater_parser)

    pipeline_image_generater_parser = model_parsers.add_parser(name="pipeline_image_generater")
    pipeline_image_generater_parser_builder(pipeline_image_generater_parser)

    joint_generater_parser = model_parsers.add_parser(name="joint_generater")
    joint_generater_parser_builder(joint_generater_parser)


def pipeline_text_generater_parser_builder(parser: ArgumentParser) -> None:
    parser.add_argument("--llm_dir", default="../llm/Llama-2-7b-chat-hf", type=str, required=True, dest="llm_dir")
    parser.add_argument("--max_new_tokens", default=30, type=int, required=True, dest="max_new_tokens")
    parser.add_argument("--num_beams", default=2, type=int, required=True, dest="num_beams")


def pipeline_image_generater_parser_builder(parser: ArgumentParser) -> None:
    parser.add_argument("--diffusion_dir", default="../llm/stable-diffusion-2-1-base", type=str, required=True, dest="diffusion_dir")


def joint_generater_parser_builder(parser: ArgumentParser) -> None:
    parser.add_argument("--llm_dir", default="../llm/Llama-2-7b-chat-hf", type=str, required=True, dest="llm_dir")
    parser.add_argument("--max_new_tokens", default=30, type=int, required=True, dest="max_new_tokens")
    parser.add_argument("--num_beams", default=2, type=int, required=True, dest="num_beams")
    parser.add_argument("--diffusion_dir", default="../llm/stable-diffusion-2-1-base", type=str, required=True, dest="diffusion_dir")
