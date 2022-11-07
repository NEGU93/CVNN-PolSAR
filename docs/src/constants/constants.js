export const DATASET_OPTIONS = ["Flevoland", "SF-AIRSAR", "Oberpfaffenhofen"];
export const SUBSET_OPTIONS = ["test", "train", "validation"];
export const METRIC_OPTIONS = ["loss", "accuracy", "average_accuracy"];

export const HTML_FILES_NAMES = [
  "box-plot.html",
  "histogram.html",
  "violin-plot.html",
  "lines-plot.html",
  "per-class-bar.html"
];
export const PNG_FILES_NAMES = ["ground_truth.png", "PauliRGB.bmp"];
export const PUBLICATION_FILES = {
  "Flevoland": {
    "authors": "J. A. Barrachina, C. Ren, C. Morisseau, G. Vieillard and J.-P. Ovarlez",
    "title": "Merits of Complex-Valued Neural Networks for PolSAR image segmentation.",
    "published": "GRETSI - XXVIIIème Colloque Francophone de Traitement du Signal et des Images",
    "year": 2022
  },
  "SF-AIRSAR": {
    "authors": "J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau and J.-P. Ovarlez",
    "title": "Merits of Complex-Valued Neural Networks for PolSAR image segmentation.",
    "published": "ICIP - IEEE International Conference in Image Processing",
    "year": 2022
  },
  "Oberpfaffenhofen": {
    "authors": "J. A. Barrachina, C. Ren, C. Morisseau, G. Vieillard and J.-P. Ovarlez",
    "title": "Comparison between equivalent architectures of complex-valued and real-valued neural networks - Application on Polarimetric SAR image segmentation.",
    "published": "JSPS – Springer Journal of Signal Processing Systems for Signal, Image, and Video Technology special issue on IEEE MLSP",
    "year": 2022
  },
}
