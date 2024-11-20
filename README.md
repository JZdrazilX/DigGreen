# Next-generation high-throughput phenotyping with trait prediction through adaptable multi-task computational intelligence
## About:
Phenotypes, defining an organism's behaviour and physical attributes, arise from the
complex, dynamic interplay of genetics, development, and environment, whose interactions
make it enormously challenging to forecast future phenotypic traits of a plant at a given
moment. This work reports AMULET, a modular approach that uses imaging-based high-
throughput phenotyping and machine learning to predict morphological and physiological
plant traits hours to days before they are visible. AMULET streamlines the phenotyping
process by integrating plant detection, prediction, segmentation, and data analysis, enhancing
workflow efficiency and reducing time. The machine learning models used data from over
30,000 plants, using the Arabidopsis thaliana-Pseudomonas syringae pathosystem.
AMULET also demonstrated its adaptability by accurately detecting and predicting
phenotypes of in vitro potato plants after minimal fine-tuning with a small dataset. The
general approach implemented through AMULET streamlines phenotyping and will improve
breeding programs and agricultural management by enabling pre-emptive interventions
optimising plant health and productivity.

![pipe](https://github.com/user-attachments/assets/d453ec31-9e8d-4b6b-8bda-d1a7d6d819dd)

## Overview of Project Blocks:

### Block (1): Object Detection
In this phase, we deploy an object detection model to accurately localize Arabidopsis Thaliana plants in various plate configurations. This model helps us identify and track the position of plants throughout our experiments.

### Block (2): Image Prediction
Our image prediction module uses data from two previous timestamps to predict the future appearance of the plants. This predictive model aids in understanding plant growth patterns and potential developmental changes.

### Block (3): Segmentation
Segmentation is crucial for isolating the plants from their background, effectively focusing analysis on the plant itself. This step ensures that our models analyze only the relevant plant data, free from any background noise.

### Block (4): Health Estimation
After segmentation, this block estimates various descriptors from the plants, providing insights into their 'health' status. By analyzing these descriptors, we can assess the overall well-being and condition of the plants throughout the study.

![pip_pot](https://github.com/user-attachments/assets/3271234e-b055-4e3b-876d-ee6d8579d62b)

We understand the importance of practical applications for AI technologies, which is why part of our pipeline was specifically adapted for in vitro potato plants (see image above). This demonstrates that with minimal effort, such as fine-tuning, our approach can achieve promising results across various crop types, highlighting its versatility and potential for broader agricultural applications.


## Note on Code and Datasets Availability
Please note that this repository does not contain the complete source code or datasets of our project. It is designed primarily to showcase various components of our pipeline, such as data preparation, and usage of models. If you are interested in accessing the full code and dataset, it is available upon request. To request access, please contact us directly, providing your affiliation and the purpose for which you intend to use the code.

## Contacts
Plant oriented and dataset questions: nuria.de@upol.cz
Machine leanring oriented questions: jan.zdrazil@vsb.cz






