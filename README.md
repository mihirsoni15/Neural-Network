# Neural Network Predicting Chances of Admission at UCLA 

[Visit app here]([URL](https://neural-network-mihir.streamlit.app/))
 
## Project Overview

The goal of this project is to create a neural network model that predicts the probability of an applicant being admitted to UCLA. It is intended for educational purposes and can serve as a starting point for projects related to academic admissions, predictive analytics, and deep learning applications.

Key features include:

- **End-to-End Pipeline:** From data collection and preprocessing to model training and deployment.
- **Neural Network Implementation:** Built using deep learning frameworks to handle regression tasks.
- **Interactive Application:** A web-based interface (via Flask or Streamlit) for users to enter applicant details and view prediction outcomes.

## Dataset Description

The dataset should include applicant information relevant to UCLA’s admission process. Typical features may include:

- **GRE Score** – Graduate Record Examination score  
- **TOEFL Score** – Test of English as a Foreign Language score  
- **University Rating** – Ranking of the undergraduate institution  
- **Statement of Purpose (SOP) Rating** – Quality score of the statement of purpose  
- **Letter of Recommendation (LOR) Rating** – Average score for letters of recommendation  
- **CGPA** – Undergraduate Grade Point Average  
- **Research Experience** – Binary indicator (0 = no experience, 1 = experience)  
- **Chance of Admit** – Target variable (continuous value between 0 and 1)

Place your CSV file (e.g., `admissions_data.csv`) in the `data/` folder.

---

## Model Architecture

The neural network is designed to handle regression problems. A typical architecture may include:

- **Input Layer:** Accepts standardized numerical features.  
- **Hidden Layers:** Multiple dense layers with activation functions (e.g., ReLU) and optional dropout layers to prevent overfitting.  
- **Output Layer:** A single neuron using a sigmoid activation (if modeling probabilities) or a linear activation (if modeling continuous scores).

Hyperparameters such as the learning rate, batch size, number of epochs, and the number of neurons per layer can be adjusted in the `train.py` or within the `model_definition.py` module.

---

## Getting Started

### Prerequisites

- Python 3.7 or higher  
- Package installer `pip` or `conda`  
- Git (optional, for version control)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/Neural-Network-UCLA.git
   cd Neural-Network-UCLA

   pip install -r requirements.txt
