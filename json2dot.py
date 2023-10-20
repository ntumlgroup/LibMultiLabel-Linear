import json
import sys


def linebreaked(label: str):
    lines = []
    while len(label) > 20:
        dot = label.find('.', 15)
        if dot == -1:
            break
        lines.append(label[:dot])
        label = label[dot:]
    if len(label) > 0:
        lines.append(label)
    return "\\n".join(lines)


for path in sys.argv[1:]:
    with open(path) as f:
        obj = json.load(f)
        labels = dict()
        for k, vs in obj.items():
            labels[k.replace(".", "_")] = k
            for v in vs:
                labels[v.replace(".", "_")] = v

        print('digraph {')
        print('graph [pack=true]')
        for node, label in labels.items():
            print(f'{node} [label="{linebreaked(label)}"]')

        for k, vs in obj.items():
            print(f'\t{k} -> {{{" ".join(vs)}}}'.replace(".", "_"))
        print('}')
