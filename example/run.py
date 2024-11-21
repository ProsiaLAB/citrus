import sys

sys.path.append("..")


from citrus.citrus import run
from citrus.configure import load_config


def main():
    """
    1. Initialize `par` with default values.
    2. Create FITS object.
    3. Initialize the images with default values.
    4. Call input routines from `citrus/model.py` to set both `par` and `image` values.
    """
    par, imgs = load_config()

    # Run citrus
    run(par, imgs)


if __name__ == "__main__":
    main()
