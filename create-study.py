import optuna
import argparse
import os

from src.arguments import get_parser


parser = get_parser()

parser.add_argument('--study-name', type=str, required=True)

parser.add_argument('--optuna-sampler', type=str, default="TPESampler")
parser.add_argument('--optuna-pruner', type=str, default="SuccessiveHalvingPruner")

parser.add_argument('--optuna-replicas', type=int, default=3)


args = parser.parse_args()

study = optuna.create_study(
    study_name=args.study_name,
    direction="maximize",
    sampler=getattr(optuna.samplers, args.optuna_sampler)(),
    pruner=getattr(optuna.pruners, args.optuna_pruner)(),
    storage=os.environ["OPTUNA_STORAGE"]
)

study.set_user_attr("args", vars(args))
