#  SmartEventPlanner

**SmartEventPlanner** is a Python-based web application designed to simplify event planning and estimate event costs using Machine Learning. It provides an interactive interface for planning events, exploring data, and predicting estimated expenses.

##  Features

-  **Event Cost Prediction** – Predict estimated event costs using a trained Machine Learning model.
-  **Event Planning** – Organize and manage event-related information.
-  **Vendor Data** – Store and explore vendor-related information.
-  **Machine Learning** – Uses Scikit-learn for model training and prediction.
-  **Data Visualization** – Visualize event data and insights using Matplotlib.
-  **Interactive Web Application** – Built using Streamlit.
-  **CSV Data Management** – Uses CSV files for storing datasets.

##  Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

##  Project Structure

```text
SmartEventPlanner/
│
├── data/
│   ├── events.csv
│   └── vendors.csv
│
├── model/
│
├── models/
│   ├── evaluation_metrics.json
│   └── event_cost_model.pkl
│
├── app.py
├── generate_data.py
├── train_model.py
├── requirements.txt
└── README.md
```

##  Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/SmartEventPlanner.git
```

### 2. Navigate to the project directory

```bash
cd SmartEventPlanner
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at:

http://localhost:8501

## Machine Learning Model

The project uses a Machine Learning model to estimate event costs based on input data.

The model training script is:

```bash
python train_model.py
```

The trained model is saved and used by the application for predictions.

##  Dataset

The project includes CSV datasets:

- `events.csv` – Event-related data.
- `vendors.csv` – Vendor-related data.

These datasets are used for data processing, model training, and application functionality.

##  Requirements

The required Python libraries are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt
```

##  Project Objective

The main objective of SmartEventPlanner is to simplify event planning and provide estimated event budgets through Machine Learning, helping users make more informed decisions during event organization.

##  Future Improvements

- Real-time vendor information
- Online event booking
- User authentication
- Event budget optimization
- Cloud database integration
- Deployment as a public web application
- Improved prediction accuracy with real-world datasets

##  Author

Fathima Sinfa

---

⭐ If you find this project interesting, feel free to star the repository!
