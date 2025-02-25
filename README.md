# Term Deposit Marketing

## Background
We are a small startup providing machine learning solutions in the European banking market, focusing on fraud detection, sentiment classification, and customer intention prediction. Our goal is to improve the success rate of customer calls for product subscriptions by developing an evolving machine learning system that offers high success outcomes while maintaining interpretability.

## Data Description
The dataset comes from direct marketing efforts of a European bank. The campaign involved calling customers multiple times to promote term deposit subscriptions. All personally identifiable information has been removed for privacy.

### Attributes
- `age`: Age of customer (numeric)
- `job`: Type of job (categorical)
- `marital`: Marital status (categorical)
- `education`: Education level (categorical)
- `default`: Has credit in default? (binary)
- `balance`: Average yearly balance in euros (numeric)
- `housing`: Has a housing loan? (binary)
- `loan`: Has personal loan? (binary)
- `contact`: Contact communication type (categorical)
- `day`: Last contact day of the month (numeric)
- `month`: Last contact month of the year (categorical)
- `duration`: Last contact duration in seconds (numeric)
- `campaign`: Number of contacts during this campaign (numeric)

### Target Variable
- `y`: Has the client subscribed to a term deposit? (binary: yes/no)

### Download Data
[Dataset Link](https://drive.google.com/file/d/1EW-XMnGfxn-qzGtGPa3v_C63Yqj2aGf7)

## Goals and Success Metrics
- Develop a model to predict whether a customer will subscribe to a term deposit.
- Improve call success rates through data-driven decision-making.
- Provide interpretability for clients to understand model outcomes.

## Project Structure
```
└── term_deposit_marketing/
    ├── data/
    │   └── raw/
    ├── models/
    │   ├── layer1_model.pkl
    │   └── layer2_model.pkl
    ├── notebooks/
    │   ├── 01_Exploratory_Data_Analysis.ipynb 
    │   ├── 02_Model_Benchmarking.ipynb
    │   └── 03_Customer_Targeting_Pipeline.ipynb
    ├── src/
    │   ├── __init__.py
    │   ├── data_exploration.py
    │   ├── data_feature_importance.py
    │   ├── data_hypertuning.py
    │   └── data_manipulation.py
    ├── requirements.txt
    ├── README.md
```

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/clturner/Term_Deposit_Marketing.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Term_Deposit_Marketing
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
