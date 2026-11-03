# Demo video shot list (under 3 minutes)

Record the live Streamlit app (or `streamlit run streamlit_app.py` locally). No music. Captions must use
numbers that are on screen. The figures quoted below come from `artifacts/metrics.json`.

| Time | On screen | Say / caption |
|---|---|---|
| 0:00–0:15 | App header with the headline line | "Most pneumonia models give you a score and stop. This one tells you when it is unsure. On 624 held-out pediatric X-rays, all 135 of its "likely normal" answers were right; it referred 52 to a radiologist." |
| 0:15–0:45 | Pick example 1 (confident normal), then 3 (confident pneumonia) | Read the band and the calibrated chance out loud. Point at the heatmap: "this is where the evidence for that call is." |
| 0:45–1:15 | Pick an "Unsure case" example | "Here it is between 20% and 80%, so it does not guess: it refers. That band was fixed before testing." |
| 1:15–1:40 | Pick "Its most confident mistake" | "And here is its worst mistake on the test set. Showing it is the point: no model is right every time." |
| 1:40–2:20 | "How reliable is it?" tab: reliability diagram, then coverage curve | "Calibrated means when it says 80%, about 80 in 100 really are pneumonia. Here is the honest part: calibrated on validation, it is still overconfident toward pneumonia on the test split (calibration error 0.117), because the test films come from a different split. Refer the least certain cases and accuracy climbs." |
| 2:20–2:45 | "Limits and data" tab | "One pediatric dataset, 128-pixel images, two classes. Not a medical device." |
| 2:45–3:00 | GitHub README top | "Code, training job, metrics with 95% intervals, all in the repo. Problem HI-01." |
