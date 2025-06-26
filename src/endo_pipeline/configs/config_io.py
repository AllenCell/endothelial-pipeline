import yaml


def list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def yaml_encoder(data):
    yaml.SafeDumper.add_representer(list, list_representer)
    return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, width=80, indent=2)
