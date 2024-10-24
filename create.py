import os

# Define the file structure
structure = {
    "bitcoin_price_prediction": [
        "data/raw/bitcoin_data.csv",
        "models/trained_model.pkl",
        "scripts/data_preprocessing.py",
        "scripts/model_training.py",
        "scripts/live_prediction.py",
        "app.py",
    ]
}

# Function to create directories and files
def create_structure(base_path, structure):
    for folder, files in structure.items():
        for file_path in files:
            full_path = os.path.join(base_path, file_path)
            dir_name = os.path.dirname(full_path)
            
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    pass

# Create the structure
base_path = "."  # Current directory
create_structure(base_path, structure)

print("Project structure created successfully!")
