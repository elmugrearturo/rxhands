# Finger landmark labeling for left handed x-ray images

This project uses two algorithms (symbolic and neural) to extract finger landmarks from left hand x-rays. It's intended to be used as an auxiliary tool for forensics research at [UNAM](http://www.cienciaforense.facmed.unam.mx/).

## Getting Started

The latest stable version can be downloaded from the [PyPI](https://pypi.org/project/rxhands-unam-colab/). 

### Prerequisites

The code is written in Python 3, and it relies on OpenCV, SciPy and scikit-image:

* [OpenCV](https://opencv.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-image](https://scikit-image.org/)
* [TensorFlow](https://www.tensorflow.org/)

Dependencies are automatically managed by `pip`.

### Installing

To download, you can simply create a `virtualenv` and install the project with `pip`:

```
pip install rxhands-unam-colab
```

### Command-line execution

Extraction can be performed with either the symbolic algorithm or the neural one.

```
rxhands [-h] -al {symbolic,neural} [-ch] [-pre] [-of OUTPUT_FOLDER] INPUT_FOLDER
```
where:

* `-h` help
* `INPUT_FOLDER` the path to the folder where the input images are stored.
* `OUTPUT_FOLDER` the path to the folder where the results will be stored.
* `-al` can receive either `symbolic` or `neural`.
* `-pre` means that gray level normalization will be applied to the input images.
* `-ch` if present, the script will try to crop the image to contain only the hand.

### Examples

```
rxhands -al neural -ch -of results/ data/
```

This command will label landmarks through the neural algorithm. The images in folder `data/` will be cropped on input.

```
rxhands -al symbolic -of results/ data/
```

This command will label landmarks through the symbolic algorithm.

## Authors

* **Arturo Curiel** - *Initial work* - [website](https://arturocuriel.com)

See also the list of [contributors](https://github.com/forensics-colab-unam/rxhands-unam-colab/contributors) who participated in this project.

## License

This project is licensed under the GNU/GPL3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

The neural model was implemented from:

```
@inproceedings{Payer2016,
  title     = {Regressing Heatmaps for Multiple Landmark Localization Using {CNNs}},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention - {MICCAI} 2016},
  doi       = {10.1007/978-3-319-46723-8_27},
  pages     = {230--238},
  year      = {2016},
}
```
Using the [Digital Hand Atlas](https://ipilab.usc.edu/research/baaweb/) as training data.
