# Intrusion Detection System using Federated Learning, Online Learning, SDN and blockchain

## Overview
This project implements a Anomaly detection and mitigation system using a machine learning model within the Ryu SDN Controller. The system monitors network traffic, detects potential DDoS attacks, and blocks malicious traffic while allowing legitimate traffic to pass through.

## Requirements
- Linux Operating System (Note: This project is designed to work on Linux)
- Python 3.9
- Mininet
- Ryu
- Scapy
- IPFS

## Setup Instructions

### 1. Install Ryu and Mininet
Install Ryu:

```bash
sudo apt-get install ryu
pip install ryu
```
Install Mininet:
```
sudo apt-get install mininet

```
Clone the Repository
```
git clone git@github.com:MustangBro7/Federeated-Anomaly-Detection-System.git
cd Federeated-Anomaly-Detection-System
```
### Install Required Python Packages
Create a virtual environment using venv
```
python3.9 -m venv ryu-python3.9-venv
```
Activate the virtual environment
```
source ryu-python3.9-venv/bin/activate
```
Install the required Python packages listed in requirements.txt
```
pip install -r requirements.txt
```
### Train the Machine Learning Model
```
python training.py
```
### Running the Application
Start the Ryu application
```
ryu-manager ddos_with_model.py
```
Open a new terminal and start Mininet with a simple topology
```
sudo -E mn --controller=remote,ip=127.0.0.1 --topo=single,4 --mac --switch=ovsk
```
### Conducting a DDoS Attack
To simulate a DDoS attack, you can use the following command from one of the hosts in Mininet (e.g., h1)


In the mininet intereface run the following command
```
python3 attack_simulation.py
```

### Notes
- Ensure you are running on a Linux environment for compatibility with Mininet.
- The provided Python script uses a machine learning model trained to detect DDoS attacks. Ensure you have the required model files in the specified paths.


## Additional Information
For more details on Ryu and Mininet, refer to their official documentation:

- [Ryu Documentation](https://ryu.readthedocs.io/en/latest/)
- [Mininet Documentation](http://mininet.org/walkthrough/)
