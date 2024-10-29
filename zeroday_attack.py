from scapy.all import *
import json
import time
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define base features for each attack type and benign traffic (25 new attacks)
base_attack_features = {
    12: {'Tot Fwd Pkts': 1, 'TotLen Fwd Pkts': 90, 'Bwd Pkt Len Max': 200, 'Flow Pkts/s': 60.5, 'Fwd IAT Mean': 500, 'Bwd IAT Tot': 300, 'Bwd IAT Mean': 250, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 12, 'Idle Max': 1200},  # SYN Flood
    13: {'Tot Fwd Pkts': 2, 'TotLen Fwd Pkts': 80, 'Bwd Pkt Len Max': 100, 'Flow Pkts/s': 0.5, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1000, 'Fwd Seg Size Min': 10, 'Idle Max': 0},  # ICMP Flood
    14: {'Tot Fwd Pkts': 10, 'TotLen Fwd Pkts': 2000, 'Bwd Pkt Len Max': 500, 'Flow Pkts/s': 5000, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 1, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 2000, 'Fwd Seg Size Min': 20, 'Idle Max': 0},  # UDP Flood
    15: {'Tot Fwd Pkts': 15, 'TotLen Fwd Pkts': 3500, 'Bwd Pkt Len Max': 1000, 'Flow Pkts/s': 3000, 'Fwd IAT Mean': 1000, 'Bwd IAT Tot': 1000, 'Bwd IAT Mean': 1000, 'RST Flag Cnt': 0, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 3000, 'Fwd Seg Size Min': 30, 'Idle Max': 2000},  # DNS Amplification
    16: {'Tot Fwd Pkts': 5, 'TotLen Fwd Pkts': 600, 'Bwd Pkt Len Max': 120, 'Flow Pkts/s': 2000, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 1, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 2048, 'Fwd Seg Size Min': 25, 'Idle Max': 1500},  # NTP Amplification
    17: {'Tot Fwd Pkts': 12, 'TotLen Fwd Pkts': 7000, 'Bwd Pkt Len Max': 1000, 'Flow Pkts/s': 10000, 'Fwd IAT Mean': 5000, 'Bwd IAT Tot': 5000, 'Bwd IAT Mean': 5000, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 512, 'Fwd Seg Size Min': 20, 'Idle Max': 100},  # HTTP Flood
    18: {'Tot Fwd Pkts': 20, 'TotLen Fwd Pkts': 1200, 'Bwd Pkt Len Max': 120, 'Flow Pkts/s': 6000, 'Fwd IAT Mean': 1000, 'Bwd IAT Tot': 1000, 'Bwd IAT Mean': 1000, 'RST Flag Cnt': 0, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 5120, 'Fwd Seg Size Min': 50, 'Idle Max': 5000},  # Ping of Death
    19: {'Tot Fwd Pkts': 3, 'TotLen Fwd Pkts': 640, 'Bwd Pkt Len Max': 200, 'Flow Pkts/s': 300, 'Fwd IAT Mean': 600, 'Bwd IAT Tot': 300, 'Bwd IAT Mean': 100, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 16, 'Idle Max': 200},  # Smurf Attack
    20: {'Tot Fwd Pkts': 25, 'TotLen Fwd Pkts': 7500, 'Bwd Pkt Len Max': 500, 'Flow Pkts/s': 1500, 'Fwd IAT Mean': 0, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 2560, 'Fwd Seg Size Min': 15, 'Idle Max': 500},  # ARP Spoofing
    21: {'Tot Fwd Pkts': 2, 'TotLen Fwd Pkts': 400, 'Bwd Pkt Len Max': 500, 'Flow Pkts/s': 400, 'Fwd IAT Mean': 200, 'Bwd IAT Tot': 100, 'Bwd IAT Mean': 50, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1280, 'Fwd Seg Size Min': 40, 'Idle Max': 300},  # Teardrop Attack
    22: {'Tot Fwd Pkts': 1, 'TotLen Fwd Pkts': 120, 'Bwd Pkt Len Max': 500, 'Flow Pkts/s': 2500, 'Fwd IAT Mean': 1000, 'Bwd IAT Tot': 500, 'Bwd IAT Mean': 500, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 512, 'Fwd Seg Size Min': 10, 'Idle Max': 1000},  # DNS Flood
    23: {'Tot Fwd Pkts': 6, 'TotLen Fwd Pkts': 300, 'Bwd Pkt Len Max': 200, 'Flow Pkts/s': 1200, 'Fwd IAT Mean': 500, 'Bwd IAT Tot': 200, 'Bwd IAT Mean': 100, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 8, 'Idle Max': 500},  # IP Fragmentation Attack
    24: {'Tot Fwd Pkts': 2, 'TotLen Fwd Pkts': 80, 'Bwd Pkt Len Max': 120, 'Flow Pkts/s': 12000, 'Fwd IAT Mean': 600, 'Bwd IAT Tot': 300, 'Bwd IAT Mean': 150, 'RST Flag Cnt': 0, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 10240, 'Fwd Seg Size Min': 30, 'Idle Max': 2000},  # Man-in-the-Middle Attack
    25: {'Tot Fwd Pkts': 1, 'TotLen Fwd Pkts': 200, 'Bwd Pkt Len Max': 100, 'Flow Pkts/s': 250, 'Fwd IAT Mean': 600, 'Bwd IAT Tot': 300, 'Bwd IAT Mean': 150, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 12, 'Idle Max': 200},  # Evil Twin Attack
    26: {'Tot Fwd Pkts': 10, 'TotLen Fwd Pkts': 400, 'Bwd Pkt Len Max': 120, 'Flow Pkts/s': 500, 'Fwd IAT Mean': 200, 'Bwd IAT Tot': 400, 'Bwd IAT Mean': 200, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1280, 'Fwd Seg Size Min': 8, 'Idle Max': 500},  # Rogue Access Point Attack
    27: {'Tot Fwd Pkts': 5, 'TotLen Fwd Pkts': 600, 'Bwd Pkt Len Max': 200, 'Flow Pkts/s': 1000, 'Fwd IAT Mean': 300, 'Bwd IAT Tot': 500, 'Bwd IAT Mean': 250, 'RST Flag Cnt': 1, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 2048, 'Fwd Seg Size Min': 25, 'Idle Max': 500},  # Cross-Site Scripting (XSS)
    28: {'Tot Fwd Pkts': 7, 'TotLen Fwd Pkts': 500, 'Bwd Pkt Len Max': 300, 'Flow Pkts/s': 2000, 'Fwd IAT Mean': 400, 'Bwd IAT Tot': 100, 'Bwd IAT Mean': 50, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1500, 'Fwd Seg Size Min': 10, 'Idle Max': 300},  # SQL Slammer Worm
    29: {'Tot Fwd Pkts': 3, 'TotLen Fwd Pkts': 800, 'Bwd Pkt Len Max': 500, 'Flow Pkts/s': 1200, 'Fwd IAT Mean': 600, 'Bwd IAT Tot': 300, 'Bwd IAT Mean': 150, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 2048, 'Fwd Seg Size Min': 30, 'Idle Max': 200},  # Social Engineering Attack
    30: {'Tot Fwd Pkts': 6, 'TotLen Fwd Pkts': 1000, 'Bwd Pkt Len Max': 300, 'Flow Pkts/s': 500, 'Fwd IAT Mean': 1000, 'Bwd IAT Tot': 500, 'Bwd IAT Mean': 250, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 20, 'Idle Max': 500},  # DNS Spoofing
    31: {'Tot Fwd Pkts': 8, 'TotLen Fwd Pkts': 800, 'Bwd Pkt Len Max': 300, 'Flow Pkts/s': 3000, 'Fwd IAT Mean': 300, 'Bwd IAT Tot': 300, 'Bwd IAT Mean': 150, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 2560, 'Fwd Seg Size Min': 15, 'Idle Max': 500},  # XML Injection
    32: {'Tot Fwd Pkts': 4, 'TotLen Fwd Pkts': 400, 'Bwd Pkt Len Max': 200, 'Flow Pkts/s': 1000, 'Fwd IAT Mean': 500, 'Bwd IAT Tot': 200, 'Bwd IAT Mean': 100, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1024, 'Fwd Seg Size Min': 16, 'Idle Max': 200},  # LDAP Injection
    33: {'Tot Fwd Pkts': 3, 'TotLen Fwd Pkts': 600, 'Bwd Pkt Len Max': 500, 'Flow Pkts/s': 1500, 'Fwd IAT Mean': 400, 'Bwd IAT Tot': 200, 'Bwd IAT Mean': 100, 'RST Flag Cnt': 0, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 1500, 'Fwd Seg Size Min': 20, 'Idle Max': 300},  # Zero-Day Exploit
    34: {'Tot Fwd Pkts': 7, 'TotLen Fwd Pkts': 700, 'Bwd Pkt Len Max': 400, 'Flow Pkts/s': 2000, 'Fwd IAT Mean': 300, 'Bwd IAT Tot': 200, 'Bwd IAT Mean': 100, 'RST Flag Cnt': 0, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 1000, 'Fwd Seg Size Min': 10, 'Idle Max': 400},  # Password Cracking Attack
    35: {'Tot Fwd Pkts': 6, 'TotLen Fwd Pkts': 1200, 'Bwd Pkt Len Max': 600, 'Flow Pkts/s': 1500, 'Fwd IAT Mean': 200, 'Bwd IAT Tot': 200, 'Bwd IAT Mean': 100, 'RST Flag Cnt': 1, 'URG Flag Cnt': 0, 'Init Fwd Win Byts': 2048, 'Fwd Seg Size Min': 15, 'Idle Max': 300},  # DNS Tunnel Attack
    36: {'Tot Fwd Pkts': 12, 'TotLen Fwd Pkts': 600, 'Bwd Pkt Len Max': 300, 'Flow Pkts/s': 2500, 'Fwd IAT Mean': 500, 'Bwd IAT Tot': 200, 'Bwd IAT Mean': 100, 'RST Flag Cnt': 0, 'URG Flag Cnt': 1, 'Init Fwd Win Byts': 1000, 'Fwd Seg Size Min': 30, 'Idle Max': 200},  # ARP Poisoning
}

# Map attack IDs to names for logging purposes
attack_names = {
    12: 'SYN Flood',
    13: 'ICMP Flood',
    14: 'UDP Flood',
    15: 'DNS Amplification',
    16: 'NTP Amplification',
    17: 'HTTP Flood',
    18: 'Ping of Death',
    19: 'Smurf Attack',
    20: 'ARP Spoofing',
    21: 'Teardrop Attack',
    22: 'DNS Flood',
    23: 'IP Fragmentation Attack',
    24: 'Man-in-the-Middle Attack',
    25: 'Evil Twin Attack',
    26: 'Rogue Access Point Attack',
    27: 'Cross-Site Scripting (XSS)',
    28: 'SQL Slammer Worm',
    29: 'Social Engineering Attack',
    30: 'DNS Spoofing',
    31: 'XML Injection',
    32: 'LDAP Injection',
    33: 'Zero-Day Exploit',
    34: 'Password Cracking Attack',
    35: 'DNS Tunnel Attack',
    36: 'ARP Poisoning',
}

# Function to generate random features for an attack based on the base features
def generate_random_features(attack_id):
    features = base_attack_features[attack_id].copy()

    # Randomize certain features within a realistic range
    if attack_id == 12:  # SYN Flood
        features['TotLen Fwd Pkts'] = random.randint(70, 120)
        features['Flow Pkts/s'] = random.uniform(50, 100)
    elif attack_id in {13, 14}:  # Flood attacks
        features['Flow Pkts/s'] = random.uniform(2000000, 4000000)
    elif attack_id in {19, 20}:  # Spoofing attacks
        features['TotLen Fwd Pkts'] = random.randint(100, 200)
        features['Flow Pkts/s'] = random.uniform(500, 1500)
    elif attack_id in {25, 26}:  # Man-in-the-Middle and Rogue AP
        features['Fwd IAT Mean'] = random.uniform(500, 2000)
    else:
        features['TotLen Fwd Pkts'] = random.randint(50, 150)
        features['Flow Pkts/s'] = random.uniform(100, 1000)

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
        12: {'protocol': 'tcp', 'repeat': 10, 'interval': 2},  # SYN Flood
        13: {'protocol': 'tcp', 'repeat': 30, 'interval': 0.5},  # ICMP Flood
        14: {'protocol': 'udp', 'repeat': 40, 'interval': 0.3},  # UDP Flood
        15: {'protocol': 'tcp', 'repeat': 100, 'interval': 0.05},  # DNS Amplification
        16: {'protocol': 'udp', 'repeat': 50, 'interval': 0.01},  # NTP Amplification
        17: {'protocol': 'tcp', 'repeat': 25, 'interval': 1},  # HTTP Flood
        18: {'protocol': 'tcp', 'repeat': 15, 'interval': 2},  # Ping of Death
        19: {'protocol': 'tcp', 'repeat': 20, 'interval': random.uniform(1, 5)},  # Smurf Attack
        20: {'protocol': 'udp', 'repeat': 20, 'interval': 2},  # ARP Spoofing
        21: {'protocol': 'tcp', 'repeat': 5, 'interval': 15},  # Teardrop Attack
        22: {'protocol': 'udp', 'repeat': 10, 'interval': 8},  # DNS Flood
        23: {'protocol': 'tcp', 'repeat': 25, 'interval': 1},  # IP Fragmentation Attack
        24: {'protocol': 'tcp', 'repeat': 15, 'interval': 2},  # Man-in-the-Middle Attack
        25: {'protocol': 'tcp', 'repeat': 30, 'interval': 0.5},  # Evil Twin Attack
        26: {'protocol': 'tcp', 'repeat': 40, 'interval': 0.3},  # Rogue Access Point Attack
        27: {'protocol': 'tcp', 'repeat': 25, 'interval': 1},  # Cross-Site Scripting (XSS)
        28: {'protocol': 'tcp', 'repeat': 10, 'interval': 2},  # SQL Slammer Worm
        29: {'protocol': 'udp', 'repeat': 30, 'interval': 0.5},  # Social Engineering Attack
        30: {'protocol': 'tcp', 'repeat': 20, 'interval': random.uniform(1, 5)},  # DNS Spoofing
        31: {'protocol': 'tcp', 'repeat': 25, 'interval': 1},  # XML Injection
        32: {'protocol': 'tcp', 'repeat': 15, 'interval': 2},  # LDAP Injection
        33: {'protocol': 'tcp', 'repeat': 40, 'interval': 0.3},  # Zero-Day Exploit
        34: {'protocol': 'tcp', 'repeat': 25, 'interval': 1},  # Password Cracking Attack
        35: {'protocol': 'tcp', 'repeat': 15, 'interval': 2},  # DNS Tunnel Attack
        36: {'protocol': 'tcp', 'repeat': 20, 'interval': random.uniform(1, 5)},  # ARP Poisoning
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
    # for i in range(2):
        # logging.info(f"Starting iteration {i+1}/100")
        # simulate_attacks(src_ip, dst_ip)
        # logging.info(f"Iteration {i+1}/100 completed. Pausing before next iteration.")
        # time.sleep(5)  # Delay between iterations to mimic real-world timing
    simulate_attacks(src_ip, dst_ip)
