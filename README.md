# The Digital Greenhouse: Harnessing AI for Advanced Plant Growth Monitoring and Prediction
## About:
The "digital greenhouse" represents a pioneering approach in precision agriculture that redefines plant growth management through an AI-driven pipeline. This study takes place in an advanced laboratory environment where plant cultivation is carefully carried out under controlled conditions, ensuring the reliability and accuracy of the data collection. Our methodology integrates machine learning processes, including object detection, image time series prediction, image segmentation, plant descriptor estimation, and comprehensive result validation. The aim is to transcend mere observation and venture into proactive prediction and influence of plant growth trajectories. We extrapolate future growth patterns through sequential plant imaging and offer predictive insights into plant health and morphology. Segmentation and detailed analysis of plant images yield comprehensive descriptors that provide a deep understanding of plant health. The validation phase of our methodology consolidates its robustness and ensures the accuracy and practical relevance of our predictive models in controlled agricultural environments. This research represents a significant advance in digital agriculture. It provides practitioners with a sophisticated and comprehensive set of instruments to better monitor and predict plant growth, setting a new benchmark for technology-enabled cultivation. 

![pipe](https://github.com/user-attachments/assets/d453ec31-9e8d-4b6b-8bda-d1a7d6d819dd)

<img width="2857" alt="pipeline" src="https://github.com/JZdrazilX/DigGreen/assets/91844670/407b5bab-24b3-4287-a88b-0e52747e523c">

## Overview of Project Blocks:
### Block (A): Data Preparation
This block covers the initial stages of our project pipeline, focusing on data gathering, preprocessing, and augmentation. Here, we collect and refine the data needed for our models, ensuring it's optimized for further analysis and processing.

### Block (B): Object Detection
In this phase, we deploy an object detection model to accurately localize Arabidopsis Thaliana plants in various plate configurations. This model helps us identify and track the position of plants throughout our experiments.

### Block (C): Image Prediction
Our image prediction module uses data from two previous timestamps to predict the future appearance of the plants. This predictive model aids in understanding plant growth patterns and potential developmental changes.

### Block (D): Segmentation
Segmentation is crucial for isolating the plants from their background, effectively focusing analysis on the plant itself. This step ensures that our models analyze only the relevant plant data, free from any background noise.

### Block (E): Health Estimation
After segmentation, this block estimates various descriptors from the plants, providing insights into their 'health' status. By analyzing these descriptors, we can assess the overall well-being and condition of the plants throughout the study.

## Note on Code and Datasets Availability
Please note that this repository does not contain the complete source code or datasets of our project. It is designed primarily to showcase various components of our pipeline, such as data preparation, augmentation, and model training. If you are interested in accessing the full code, it is available upon request. To request access, please contact us directly, providing your affiliation and the purpose for which you intend to use the code.

## Contacts
Machine leanring oriented questions: jan.zdrazil@vsb.cz

Plant oriented questions: nuria.de@upol.cz





