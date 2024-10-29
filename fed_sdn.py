import sys
import pickle
import numpy as np
import pandas as pd
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
import json
import logging
import joblib
import os
import subprocess
import requests

# Constants
HASH_FILE_PATH = 'model_hash.txt'  # Path to the hash file
MODEL_FILE_PATH = 'rf_model.joblib'  # Local path for the RF model
FED_SERVER_IP = '127.0.0.1'  # IP address of the federated server
FED_SERVER_PORT = 8000  # Port of the federated server
GLOBAL_MODEL_PATH = 'global_model.joblib'
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

        # Load models from IPFS using the hash from the file
        self.rf_model = self.load_model_from_ipfs()
        self.xgb_model = joblib.load('xgb_model.joblib')
        self.orf_model = joblib.load('orf_model.joblib')
        self.ht_model = joblib.load('ht_model.joblib')
        self.scaler = joblib.load('scaler.joblib')
        
        self.features = [
            'Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Bwd Pkt Len Max', 'Flow Pkts/s',
            'Fwd IAT Mean', 'Bwd IAT Tot', 'Bwd IAT Mean', 'RST Flag Cnt',
            'URG Flag Cnt', 'Init Fwd Win Byts', 'Fwd Seg Size Min', 'Idle Max'
        ]
        
        # Buffers for incremental learning and replay
        self.local_data_buffer_rf = []
        self.local_data_buffer_xgb = []
        self.local_data_buffer_size = 1000

    def load_model_from_ipfs(self, model_hash=None):
        """Load the model from IPFS using the provided hash or the genesis hash."""
        if not model_hash:
            # Use the genesis hash if no model_hash provided
            if os.path.exists(HASH_FILE_PATH):
                with open(HASH_FILE_PATH, 'r') as f:
                    model_hash = f.read().strip()
            else:
                self.logger.error("Genesis hash file not found.")
                return joblib.load(MODEL_FILE_PATH)

        # Retrieve and load the model using IPFS hash
        os.system(f"ipfs cat {model_hash} > {MODEL_FILE_PATH}")
        return joblib.load(MODEL_FILE_PATH)

    def update_model(self, new_hash):
        """Update the current model with a new model from IPFS using the new hash."""
        self.rf_model = self.load_model_from_ipfs(new_hash)
        self.logger.info(f"Model updated to the latest hash: {new_hash}")

    # Define a new endpoint to handle hash updates from the server
    # @app.route('/update_model_hash', methods=['POST'])
    # def update_model_hash():
        # data = request.get_json()
        # new_hash = data.get("ipfs_cid")
        # if new_hash:
            # SimpleSwitch13.update_model(new_hash)
            # return jsonify({"message": "Model updated"}), 200
        # else:
            # return jsonify({"error": "Invalid hash data"}), 400

    # Remaining packet handling and prediction code remain unchanged

    def extract_features(self, payload):
        try:
            features = json.loads(payload.decode('utf-8'))
            if all(feature in features for feature in self.features):
                return features
            else:
                self.logger.error("Some features are missing from the payload.")
                return None
        except Exception as e:
            self.logger.error(f"Error extracting features from packet: {e}")
            return None

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info("Switch setup complete. Prediction enabled.")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id, priority=priority,
                                    match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid][dst] if dst in self.mac_to_port[dpid] else ofproto.OFPP_FLOOD
        actions = [parser.OFPActionOutput(out_port)]

        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        raw_payload = pkt[-1] if (tcp_pkt or udp_pkt) and isinstance(pkt[-1], bytes) else None

        if raw_payload:
            features = self.extract_features(raw_payload)
            if features:
                features_df = pd.DataFrame([features], columns=self.features)
                # print(features_df)
                try:
                    rf_pred = self.rf_model.predict(features_df)[0]
                    xgb_pred = self.xgb_model.predict(features_df)[0]
                    orf_pred = self.orf_model.predict_one(features)
                    ht_pred = self.ht_model.predict_one(features)
                    self.logger.info(f"RF {rf_pred} XGB {xgb_pred} HT {ht_pred} ORF {orf_pred}")

                    if rf_pred != 1 or xgb_pred != 0 or orf_pred != 1 or ht_pred != 1:
                        self.logger.info(f"Attack detected from {src}, blocking traffic.")
                        self.ht_model.learn_one(features, max(rf_pred, xgb_pred, ht_pred, orf_pred))
                        self.orf_model.learn_one(features, max(rf_pred, xgb_pred, ht_pred, orf_pred))
                        final_pred = max(rf_pred, xgb_pred, ht_pred, orf_pred)
                        final_pred = xgb_pred if final_pred == xgb_pred else final_pred - 1
                        self.local_data_buffer_rf.append((features_df, max(rf_pred, xgb_pred, ht_pred, orf_pred)))
                        self.local_data_buffer_xgb.append((features_df, final_pred))

                        if len(self.local_data_buffer_rf) >= self.local_data_buffer_size:
                            X_rf, y_rf = zip(*self.local_data_buffer_rf)
                            X_rf = pd.concat(X_rf)
                            self.rf_model.fit(X_rf, y_rf)
                            self.send_model_to_server(self.rf_model, 'rf')
                            self.local_data_buffer_rf = []
                            self.get_hash()

                    else:
                        self.logger.info(f"Normal traffic from {src}.")
                        self.ht_model.learn_one(features, 1)
                        self.orf_model.learn_one(features, 1)
                        self.local_data_buffer_rf.append((features_df, 1))
                        self.local_data_buffer_xgb.append((features_df, 0))

                except Exception as e:
                    self.logger.error(f"Error in prediction: {e}")
            else:
                self.logger.error("No valid feature data received in packet.")

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def send_model_to_server(self, model, model_type):
        try:
            model_filename = f"/tmp/{model_type}_model.joblib"
            joblib.dump(model, model_filename)
            files = {'model': open(model_filename, 'rb')}
            data = {'model_type': model_type}
            response = requests.post(f'http://{FED_SERVER_IP}:{FED_SERVER_PORT}/update_model', files=files, data=data)
            if response.status_code == 200:
                self.logger.info(f"{model_type} model file sent successfully to the server.")
            else:
                self.logger.error(f"Failed to send {model_type} model file to the server. Status Code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error while sending {model_type} model file to server: {e}")
        
        finally:
            if os.path.exists(model_filename):
                os.remove(model_filename)
    def get_hash(self):
        try:
            ipfs_hash = requests.get(f'http://{FED_SERVER_IP}:{FED_SERVER_PORT}/get_current_hash')
            if ipfs_hash.status_code == 200:	
                data = ipfs_hash.json()
                curr = data.get('ipfs_cid')
                print("New IPFS Hash",curr)
                self.load_model_from_ipfs(curr)
        except Exception as e:
            print(f"error : {e}")
