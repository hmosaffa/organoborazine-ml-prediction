Organoborazine ML Prediction
This project uses machine learning to predict the maximum absorption wavelength (λ_max) of organoborazines based on their structural and chemical properties. We employ a Random Forest Regressor model, optimized through hyperparameter tuning.
Project Overview
Organoborazines are an important class of compounds with potential applications in optoelectronic devices. This project aims to accelerate the discovery and design of new organoborazines by predicting their λ_max values using machine learning techniques.
Features

Data preprocessing and feature scaling
Random Forest Regression model
Hyperparameter tuning using GridSearchCV
Model evaluation using various metrics (MSE, MAE, R2, RMSE, Bias, Correlation Coefficient)

Installation

Clone this repository:
Copygit clone https://github.com/yourusername/organoborazine-ml-prediction.git
cd organoborazine-ml-prediction

Create a virtual environment and activate it:
Copypython -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

Install the required packages:
Copypip install -r requirements.txt


Usage
To run the model:
Copypython main.py
This will load the data, preprocess it, train the model, and print the best parameters and evaluation metrics.
Data
The dataset used in this project is stored in data/bor.xlsx. It contains various features of organoborazines and their corresponding λ_max values. Due to the size and potential confidentiality of the data, it is not included in this repository. Please contact the authors for information on accessing or generating similar data.
Model
We use a Random Forest Regressor with the following hyperparameters tuned via GridSearchCV:

n_estimators
max_depth
min_samples_split
min_samples_leaf
bootstrap

The model is trained on 80% of the data and evaluated on the remaining 20%.
Results
[Include your best results here, such as the best hyperparameters found and the model's performance metrics on the test set]
Contributing
We welcome contributions to this project. Please see the CONTRIBUTING.md file for guidelines on how to contribute.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For any questions or concerns, please open an issue on this repository or contact the authors directly.
