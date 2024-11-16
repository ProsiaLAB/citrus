import tomllib

from . import datastructures as ds


def load_config():
    with open("input.toml", "r") as f:
        config = tomllib.loads(f.read())

    par = ds.init_input_parameters()
    for key, value in config["parameters"].items():
        par[key] = value

    # Get number of images
    # Images are with TOML sections [image-##]
    nimages = 0
    for key in config.keys():
        if key.startswith("image-"):
            nimages += 1

    # Initialize the images
    imgs = ds.init_images(nimages)

    for i in range(nimages):
        for key, value in config[f"image-{i+1:02}"].items():
            imgs[key][i] = value

    return par, imgs
