import tomllib

from . import structarrays as sa
from .datastructures import Image, InputParams


def load_config():
    with open("input.toml", "r") as f:
        config = tomllib.loads(f.read())

    par = InputParams(**config["parameters"])

    img_keys = [key for key in config.keys() if key.startswith("image-")]

    imgs = []
    nimgs = len(img_keys)

    for i in range(1, nimgs + 1):
        img = Image(**config[f"image-{i:02}"])
        imgs.append(img)

    return par, imgs


def load_config_into_arrays():
    with open("input.toml", "r") as f:
        config = tomllib.loads(f.read())

    par = sa.init_input_parameters()
    for key, value in config["parameters"].items():
        par[key] = value

    # Get number of images
    # Images are with TOML sections [image-##]
    nimages = 0
    for key in config.keys():
        if key.startswith("image-"):
            nimages += 1

    # Initialize the images
    imgs = sa.init_images(nimages)

    for i in range(nimages):
        for key, value in config[f"image-{i+1:02}"].items():
            imgs[key][i] = value

    return par, imgs
