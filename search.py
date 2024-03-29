import yaml
import argparse
from itertools import product
import os
import json

def get_combinations(config):
    network_config = {}
    network_config_fix = {}
    for key, value in config.items():
        if isinstance(value, list) and len(value) >= 2 and value[0] == "grid_search":
            network_config[key] = value[1]
    for key, value in config["network_config"].items():
        if isinstance(value, list) and len(value) >= 2 and value[0] == "grid_search":
            network_config[key] = value[1]
        else:
            network_config_fix[key] = value
    comb = [list([i[0] for i in network_config.items()])]
    comb.append(list(product(*[i[1] for i in network_config.items()])))
    comb.append(network_config_fix)
    return comb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exps', type=str)
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-t', '--template', type=str)
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-r', '--resume', action="store_true")
    parser.add_argument('--no_checkpoint', action="store_true")
    args = parser.parse_args()

    os.makedirs(f"runs/{args.exps}", exist_ok=True)

    with open(args.config) as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    
    combs = get_combinations(config) if not args.resume else json.load("status.json")

    with open(args.template) as fp:
        template = yaml.load(fp, Loader=yaml.SafeLoader)

    from torch_trainer import TorchTrainer
    from libmultilabel.common_utils import AttributeDict
    import copy
    left_combination = copy.deepcopy(combs)
    template["network_config"] = combs[2]
    template["seed"] = args.seed
    template["model_name"] = config["model_name"]
    for comb in combs[1]:
        name = []
        for i in range(len(combs[0])):
            if combs[0][i] == "learning_rate":
                template[combs[0][i]] = comb[i]
            else:
                template["network_config"][combs[0][i]] = comb[i]
            name.append(combs[0][i])
            name.append(str(comb[i]))
        template["result_dir"] = f"./runs/{args.exps}"
        template["run_name"] = ("_").join(name)
        template = AttributeDict(template)
        template.checkpoint_dir = os.path.join(template.result_dir, template.run_name)
        template.log_path = os.path.join(template.checkpoint_dir, "logs.json")
        template.predict_out_path = os.path.join(template.checkpoint_dir, "predictions.txt")
        print(template)
        trainer = TorchTrainer(config=template, save_checkpoints=not args.no_checkpoint)
        trainer.train()
        trainer.test()
        left_combination[1].remove(comb)
        print("********* Trial Done *********")
        print(f"combination: {name}")
        json.dump({'exps': args.exps,'left':left_combination}, open("status.json", "w"), indent = 4)



if __name__ == "__main__":
    main()