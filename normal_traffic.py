from scapy.all import *
import json
import time
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base features for each attack type and benign traffic
base_attack_features = {
    1: {'Tot Fwd Pkts': 1, 'TotLen Fwd Pkts': 84, 'Bwd Pkt Len Max': 166, 'Flow Pkts/s': 54.66, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 8, 'Idle Max': 0}}

# Map attack IDs to names for logging purposes
attack_names = {
    1: 'Benign'
}

# Function to generate random features for an attack based on the base features
def generate_random_features(attack_id):
    features = base_attack_features[attack_id].copy()

    # Randomize certain features within a realistic range
    if attack_id == 1:  # Benign
        features['TotLen Fwd Pkts'] = random.randint(60, 100)
        features['Flow Pkts/s'] = random.uniform(40, 70)

    # Binary flags (RST, URG) toggle randomly
    features['RST Flag Cnt'] = random.choice([0, 1])
    features['URG Flag Cnt'] = random.choice([0, 1])

    return features

# Function to simulate sending a packet with random features for the attack
def send_custom_packet(src_ip, dst_ip, protocol, attack_id, repeat=1):
    logging.info(f"Starting {attack_names[attack_id]} attack simulation (ID {attack_id})")
    
    for _ in range(repeat):
        features = generate_random_features(attack_id)
        payload = json.dumps(features).encode('utf-8')  # Convert features to JSON-encoded bytes

        # Define protocol for the packet (TCP/UDP)
        if protocol.lower() == 'tcp':
            pkt = IP(src=src_ip, dst=dst_ip) / TCP(dport=80) / Raw(load=payload)
        elif protocol.lower() == 'udp':
            pkt = IP(src=src_ip, dst=dst_ip) / UDP(dport=80) / Raw(load=payload)
        else:
            logging.error("Unsupported protocol")
            return

        send(pkt)
        logging.info(f"Packet sent for attack ID {attack_id} ({attack_names[attack_id]})")

# Function to simulate all attacks and benign traffic with randomized features
def simulate_attacks(src_ip, dst_ip):
    # Attack behaviors
    attack_behavior = {
        1: {'protocol': 'tcp', 'repeat': 10, 'interval': 2},  # Benign
    }

    # Simulate each attack type with randomized behavior
    for attack_id, behavior in attack_behavior.items():
        logging.info(f"Simulating {attack_names[attack_id]} attack (ID {attack_id})...")
        send_custom_packet(src_ip, dst_ip, behavior['protocol'], attack_id, repeat=behavior['repeat'])
        time.sleep(behavior['interval'])  # Pause between each batch of packets

if __name__ == "__main__":
    # Prompt user for IP addresses
    src_ip = input("Enter source IP address (default 10.0.0.1): ") or "10.0.0.1"
    dst_ip = input("Enter destination IP address (default 10.0.0.2): ") or "10.0.0.2"
    
    # Run the simulation 100 times to train the online models
    # for i in range(100):
    #     logging.info(f"Starting iteration {i+1}/100")
    #     simulate_attacks(src_ip, dst_ip)
    #     logging.info(f"Iteration {i+1}/100 completed. Pausing before next iteration.")
    #     time.sleep(5)  # Delay between iterations to mimic real-world timing
    simulate_attacks(src_ip, dst_ip)
