### What this is not

- **Not a medical device.** It has not been cleared, validated or tested in any clinic. It must never be used to diagnose pneumonia or to rule it out.
- **One dataset, one population.** Every image comes from PneumoniaMNIST, which its authors describe as "based on a prior dataset of 5,856 pediatric chest X-Ray images". It has never seen adult films, other hospitals, other machines, portable bedside films, or lateral views. Expect it to be wrong, and confidently wrong, on those.
- **Low resolution.** Images are center-cropped and shrunk to 128 x 128 pixels, so fine detail a radiologist would use is gone.
- **Two classes only.** "Pneumonia" here lumps bacterial and viral cases together, and "normal" means "labelled normal in this dataset". It knows nothing about other findings (effusion, tuberculosis, tubes and lines, fractures).
- **Calibration is only as good as its data.** Probabilities were calibrated on the validation split, which is cut from the same source training set; the test split has a different mix of cases (62.5% pneumonia vs 74% in validation). The reliability diagram shows how well that held up on test.
- **The heatmap is a hint, not an explanation.** A class activation map shows which regions pushed the score, at a coarse 7 x 7 grid. It can highlight text markers or borders, and a plausible-looking map does not make a prediction right.

### Data and credit

PneumoniaMNIST from MedMNIST v3.0 (Yang et al., *Scientific Data* 2023), CC BY 4.0, Zenodo record 10519652. Backbone: ResNet-18 with ImageNet weights from torchvision (BSD).

### Privacy

Uploaded images are processed in memory for one prediction and are not stored or logged by the app.
