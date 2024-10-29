from scapy.all import *
import json
import time
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base features for each attack type and benign traffic
base_attack_features = {
    1: {'Tot Fwd Pkts': 1, 'TotLen Fwd Pkts': 84, 'Bwd Pkt Len Max': 166, 'Flow Pkts/s': 54.66, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 8, 'Idle Max': 0},  # Benign
    2: {'Tot Fwd Pkts': 1, 'TotLen Fwd Pkts': 0, 'Bwd Pkt Len Max': 0, 'Flow Pkts/s': 2000000.0, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 26883, 'Fwd Seg Size Min': 40, 'Idle Max': 0},  # FTP-BruteForce
    3: {'Tot Fwd Pkts': 1, 'TotLen Fwd Pkts': 0, 'Bwd Pkt Len Max': 0, 'Flow Pkts/s': 285714.29, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 241, 'Fwd Seg Size Min': 32, 'Idle Max': 0},  # SSH-Bruteforce
    4: {'Tot Fwd Pkts': 2, 'TotLen Fwd Pkts': 0, 'Bwd Pkt Len Max': 0, 'Flow Pkts/s': 2463.05, 'Fwd IAT Mean': 812, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 32738, 'Fwd Seg Size Min': 20, 'Idle Max': 0},  # DDOS attack-HOIC
    5: {'Tot Fwd Pkts': 2, 'TotLen Fwd Pkts': 0, 'Bwd Pkt Len Max': 0, 'Flow Pkts/s': 3717.47, 'Fwd IAT Mean': 538, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 2052, 'Fwd Seg Size Min': 20, 'Idle Max': 0},  # Bot
    6: {'Tot Fwd Pkts': 2, 'TotLen Fwd Pkts': 0, 'Bwd Pkt Len Max': 0, 'Flow Pkts/s': 0.09, 'Fwd IAT Mean': 23076676, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 221, 'Fwd Seg Size Min': 32, 'Idle Max': 23076676},  # DoS-GoldenEye
    7: {'Tot Fwd Pkts': 2, 'TotLen Fwd Pkts': 16, 'Bwd Pkt Len Max': 0, 'Flow Pkts/s': 0.04, 'Fwd IAT Mean': 99999620, 'Bwd IAT Tot': 99999616, 'Bwd IAT Mean': 99999616, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 211, 'Fwd Seg Size Min': 32, 'Idle Max': 99999614},  # DoS-Slowloris
    8: {'Tot Fwd Pkts': -11259, 'TotLen Fwd Pkts': 3834016, 'Bwd Pkt Len Max': 0, 'Flow Pkts/s': 1000.33, 'Fwd IAT Mean': 999.68, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 8, 'Idle Max': 0},  # DDOS-LOIC-UDP
    9: {'Tot Fwd Pkts': 5, 'TotLen Fwd Pkts': 646, 'Bwd Pkt Len Max': 364, 'Flow Pkts/s': 1.6, 'Fwd IAT Mean': 1252033.8, 'Bwd IAT Tot': 3226, 'Bwd IAT Mean': 1613, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 8192, 'Fwd Seg Size Min': 20, 'Idle Max': 0},  # Brute Force -Web
    10: {'Tot Fwd Pkts': 203, 'TotLen Fwd Pkts': 56330, 'Bwd Pkt Len Max': 1936, 'Flow Pkts/s': 5.41, 'Fwd IAT Mean': 280986.28, 'Bwd IAT Tot': 56761420, 'Bwd IAT Mean': 551081.75, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 8192, 'Fwd Seg Size Min': 20, 'Idle Max': 0},  # Brute Force -XSS
    11: {'Tot Fwd Pkts': 5, 'TotLen Fwd Pkts': 635, 'Bwd Pkt Len Max': 4012, 'Flow Pkts/s': 1.8, 'Fwd IAT Mean': 18261.75, 'Bwd IAT Tot': 5013865, 'Bwd IAT Mean': 1671288.33, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 8192, 'Fwd Seg Size Min': 20, 'Idle Max': 0},  # SQL Injection
}

# Map attack IDs to names for logging purposes
attack_names = {
    1: 'Benign',
    2: 'FTP-BruteForce',
    3: 'SSH-Bruteforce',
    4: 'DDOS-HOIC',
    5: 'Bot',
    6: 'DoS-GoldenEye',
    7: 'DoS-Slowloris',
    8: 'DDOS-LOIC-UDP',
    9: 'Brute Force -Web',
    10: 'Brute Force -XSS',
    11: 'SQL Injection'
}

# Function to generate random features for an attack based on the base features
def generate_random_features(attack_id):
    features = base_attack_features[attack_id].copy()

    # Randomize certain features within a realistic range
    if attack_id == 1:  # Benign
        features['TotLen Fwd Pkts'] = random.randint(60, 100)
        features['Flow Pkts/s'] = random.uniform(40, 70)
    elif attack_id in {2, 3}:  # Brute Force attacks
        features['Flow Pkts/s'] = random.uniform(1000000, 3000000)
        features['Init Fwd Win Byts'] = random.randint(20000, 40000)
    elif attack_id in {4, 8}:  # DDOS attacks
        features['Flow Pkts/s'] = random.uniform(1000, 5000)
        features['Tot Fwd Pkts'] = random.randint(2, 6)
    elif attack_id in {6, 7}:  # DoS attacks
        features['Fwd IAT Mean'] = random.uniform(20000000, 30000000)
        features['Idle Max'] = random.uniform(10000000, 40000000)
    elif attack_id in {9, 10}:  # Web Attacks
        features['Tot Fwd Pkts'] = random.randint(1, 5)
        features['Flow Pkts/s'] = random.uniform(1.0, 3.0)
    else:  # Other attacks
        features['TotLen Fwd Pkts'] = random.randint(50, 100)
        features['Flow Pkts/s'] = random.uniform(100, 1000)

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
        2: {'protocol': 'tcp', 'repeat': 30, 'interval': 0.5},  # FTP-BruteForce
        3: {'protocol': 'tcp', 'repeat': 40, 'interval': 0.3},  # SSH-Bruteforce
        4: {'protocol': 'tcp', 'repeat': 100, 'interval': 0.05},  # DDOS-HOIC
        5: {'protocol': 'tcp', 'repeat': 20, 'interval': random.uniform(1, 5)},  # Bot
        6: {'protocol': 'tcp', 'repeat': 5, 'interval': 15},  # GoldenEye
        7: {'protocol': 'tcp', 'repeat': 10, 'interval': 8},  # Slowloris
        8: {'protocol': 'udp', 'repeat': 50, 'interval': 0.01},  # LOIC-UDP
        9: {'protocol': 'tcp', 'repeat': 25, 'interval': 1},  # Web Brute Force
        10: {'protocol': 'tcp', 'repeat': 15, 'interval': 2},  # XSS
        11: {'protocol': 'tcp', 'repeat': 20, 'interval': 0.7},  # SQL Injection
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