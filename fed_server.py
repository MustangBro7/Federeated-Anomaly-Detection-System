import os
import joblib
from flask import Flask, request, jsonify
import threading
import numpy as np

app = Flask(__name__)

# Constants for file paths
MODEL_FILE_PATH = '/tmp/aggregated_rf_model.joblib'
GENESIS_HASH_FILE_PATH = 'model_hash.txt'  # Genesis block hash file
BLOCKCHAIN_FILE_PATH = 'blockchain.txt'  # File to store the chain of hashes
IPFS_OUTPUT_FILE = '/tmp/ipfs_output.txt'  # Temporary file for IPFS output

# Lock to prevent simultaneous access to the model file
model_lock = threading.Lock()

global_model = None  # Initialize the global model variable
memory_replay_buffer = []  # Memory replay buffer for past data
MAX_MEMORY_SIZE = 200  # Set a limit for the memory replay buffer size
FEDPROX_MU = 0.1  # Proximal term regularization strength

# Load genesis hash if it exists, otherwise initialize it
def load_genesis_hash():
    if os.path.exists(GENESIS_HASH_FILE_PATH):
        with open(GENESIS_HASH_FILE_PATH, 'r') as f:
            return f.read().strip()
    return None

# Function to initialize the blockchain with the genesis hash if it doesn’t exist
def initialize_blockchain():
    if not os.path.exists(BLOCKCHAIN_FILE_PATH) and genesis_hash:
        with open(BLOCKCHAIN_FILE_PATH, 'w') as f:
            f.write(f"{genesis_hash}\n")

# Function to save and upload model to IPFS, appending the new hash to the blockchain
def save_and_upload_model():
    try:
        # Save the global model to a file
        joblib.dump(global_model, MODEL_FILE_PATH)
        print("Model saved locally.")

        # Run IPFS command to add the model file
        os.system(f"ipfs add {MODEL_FILE_PATH} > {IPFS_OUTPUT_FILE}")
        
        # Extract the hash from IPFS output
        with open(IPFS_OUTPUT_FILE, 'r') as f:
            output = f.readlines()
            for line in output:
                if 'added' in line:
                    model_hash = line.split()[1]
                    
                    # Append the new hash to the blockchain file
                    with open(BLOCKCHAIN_FILE_PATH, 'a') as blockchain_file:
                        blockchain_file.write(f"{model_hash}\n")
                    
                    print(f"Model uploaded to IPFS. New hash (block): {model_hash}")
                    
                    # Notify SDN of new hash
                    #notify_sdn_of_update(model_hash)
                    return model_hash
        print("Error: Hash not found in IPFS output.")
    except Exception as e:
        print(f"Error in save_and_upload_model: {e}")
    finally:
        # Clean up the IPFS output file
        if os.path.exists(IPFS_OUTPUT_FILE):
            os.remove(IPFS_OUTPUT_FILE)
    return None

def notify_sdn_of_update(new_hash):
    try:
        response = requests.post(SDN_UPDATE_URL, json={"ipfs_cid": new_hash})
        if response.status_code == 200:
            print(f"SDN notified of updated model hash: {new_hash}")
        else:
            print(f"Failed to notify SDN. Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error notifying SDN of model update: {e}")

# Function to retrieve the latest model hash (last block) from the blockchain
def get_latest_model_hash():
    if os.path.exists(BLOCKCHAIN_FILE_PATH):
        with open(BLOCKCHAIN_FILE_PATH, 'r') as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip()
    return None

def fedprox_update(global_model, local_model, mu):
    """Applies FedProx constraint to local model updates."""
    if global_model is None:
        return local_model

    updated_estimators = []
    for global_tree, local_tree in zip(global_model.estimators_, local_model.estimators_):
        try:
            # Apply FedProx proximal constraint where shapes match
            if global_tree.tree_.value.shape == local_tree.tree_.value.shape:
                proximal_update = global_tree.tree_.value + mu * (global_tree.tree_.value - local_tree.tree_.value)
                local_tree.tree_.value = proximal_update
            # Retain local tree if shapes do not match, skipping FedProx constraint for this tree
            updated_estimators.append(local_tree)
        except Exception as e:
            print(f"Skipping FedProx for a tree due to mismatch: {e}")
            updated_estimators.append(local_tree)

    local_model.estimators_ = updated_estimators
    return local_model

@app.route('/update_model', methods=['POST'])
def update_model():
    """
    Endpoint to receive model updates from clients and aggregate them.
    """
    temp_model_path = '/tmp/received_model.joblib'  # Temporary path for received model
    try:
        # Ensure 'model' is in the POST request
        if 'model' in request.files:
            model_file = request.files['model']
            
            # Save received model temporarily
            model_file.save(temp_model_path)
            node_model = joblib.load(temp_model_path)
            
            # Lock to safely update global model
            with model_lock:
                # Check if global_model exists; if not, initialize it
                global global_model
                if global_model is None:
                    global_model = node_model
                    print("Initialized global model with received model.")
                else:
                    # FedProx aggregation
                    node_model = fedprox_update(global_model, node_model, FEDPROX_MU)
                    global_model.estimators_ += node_model.estimators_

                    # Limit the number of trees in the model to a max (e.g., 100 trees)
                    if len(global_model.estimators_) > 500:
                        global_model.estimators_ = global_model.estimators_[:500]
                    print("Updated global model with FedProx.")

                # Add to memory replay buffer
                memory_replay_buffer.append(node_model)
                if len(memory_replay_buffer) > MAX_MEMORY_SIZE:
                    memory_replay_buffer.pop(0)  # Maintain buffer size limit

                # Save and upload the aggregated model, and update IPFS hash
                model_hash = save_and_upload_model()
                if model_hash:
                    print(f"Updated IPFS hash (new block): {model_hash}")
                    return jsonify({'message': 'Model updated and uploaded to IPFS', 'ipfs_cid': model_hash}), 200
                else:
                    return jsonify({'error': 'Failed to upload model to IPFS'}), 500
        else:
            return jsonify({'error': 'No model file found in request'}), 400
    except Exception as e:
        print(f"Error during model update: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@app.route('/get_current_hash', methods=['GET'])
def get_current_hash():
    """Endpoint to retrieve the latest IPFS hash (last block)."""
    current_hash = get_latest_model_hash()
    if current_hash:
        return jsonify({'ipfs_cid': current_hash}), 200
    return jsonify({'error': 'No model hash available'}), 404

if __name__ == "__main__":
    # Load the genesis hash and initialize the blockchain
    genesis_hash = load_genesis_hash()
    initialize_blockchain()
    
    app.run(host='0.0.0.0', port=8000)
