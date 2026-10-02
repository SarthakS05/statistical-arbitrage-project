import os
import math
import time
import json
import socket
import struct
import random
import threading
import requests
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertForQuestionAnswering

class S1BudgetForcingModel:
    def __init__(self, base_model, tokenizer, max_thinking_tokens=2048):
        self.model = base_model
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.end_token = "<|end_of_thinking|>"

    def apply_budget_forcing(self, prompt, min_thinking_tokens=0, extra_waits=0):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output_tokens = []
        thinking_tokens = 0
        for _ in range(self.max_thinking_tokens * 2):
            out = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=1,
                do_sample=False
            )
            token_id = out[0, -1].item()
            token_str = self.tokenizer.decode([token_id])
            output_tokens.append(token_str)
            thinking_tokens += 1
            if self.end_token in token_str:
                if thinking_tokens < min_thinking_tokens:
                    output_tokens.append(" Wait")
                    extra_waits -= 1
                    if extra_waits <= 0:
                        break
                else:
                    break
            if thinking_tokens >= self.max_thinking_tokens:
                output_tokens.append(self.end_token)
                break
            inputs["input_ids"] = out
        return "".join(output_tokens)

class BERTFineTuner:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def encode_dataset(self, texts, labels):
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    def fine_tune(self, texts, labels, output_dir="./bert_glue"):
        dataset = self.encode_dataset(texts, labels)
        class DS:
            def __init__(self, d):
                self.d = d
            def __len__(self):
                return self.d["input_ids"].shape[0]
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.d.items()}
        train_ds = DS(dataset)
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            learning_rate=5e-5,
            logging_steps=50
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=train_ds)
        trainer.train()
        return self.model

class BERTQuestionAnswering:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)

    def answer(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        out = self.model(**inputs)
        start = int(out.start_logits.argmax())
        end = int(out.end_logits.argmax()) + 1
        tokens = inputs["input_ids"][0][start:end]
        return self.tokenizer.decode(tokens)

class ARWithOpenCV:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def run(self, overlay_path=None):
        overlay = None
        if overlay_path and os.path.exists(overlay_path):
            overlay = cv2.imread(overlay_path)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None and overlay is not None:
                for c in corners:
                    pts = c[0].astype(int)
                    x_min = pts[:, 0].min()
                    x_max = pts[:, 0].max()
                    y_min = pts[:, 1].min()
                    y_max = pts[:, 1].max()
                    w = x_max - x_min
                    h = y_max - y_min
                    resized = cv2.resize(overlay, (w, h))
                    frame[y_min:y_min + h, x_min:x_min + w] = resized
            cv2.imshow("AR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

class BitTorrentClient:
    def __init__(self, torrent_path):
        with open(torrent_path, "rb") as f:
            self.meta = self._parse_torrent(f.read())
        self.peer_id = b"-PC0001-" + os.urandom(12)
        self.peers = []

    def _parse_torrent(self, data):
        return {}

    def _build_handshake(self):
        pstr = b"BitTorrent protocol"
        reserved = b"\x00" * 8
        info_hash = b"\x00" * 20
        return struct.pack("!B", len(pstr)) + pstr + reserved + info_hash + self.peer_id

    def add_peer(self, host, port):
        self.peers.append((host, port))

    def connect_and_handshake(self, peer):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(peer)
        hs = self._build_handshake()
        s.sendall(hs)
        resp = s.recv(68)
        s.close()
        return resp

class CryptoStatArb:
    def __init__(self, symbols, window_minutes=1440, horizon_minutes=120):
        self.symbols = symbols
        self.window = window_minutes
        self.horizon = horizon_minutes
        self.model_rf = RandomForestClassifier(n_estimators=100, max_depth=15)
        self.model_lr = LogisticRegression(max_iter=100)

    def fetch_data(self, symbol, limit=40000):
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit={limit}"
        r = requests.get(url)
        j = r.json()
        df = pd.DataFrame(j["Data"]["Data"])
        df["ret"] = df["close"].pct_change().fillna(0.0)
        return df

    def build_features_targets(self):
        all_feats = []
        all_labels = []
        for sym in self.symbols:
            df = self.fetch_data(sym)
            rets = df["ret"].values
            feats = []
            labels = []
            for t in range(self.window, len(rets) - self.horizon - 1):
                f = []
                for m in [20, 40, 60, 80, 100, 120]:
                    f.append(np.prod(1 + rets[t - m:t]) - 1)
                for k in range(2, 13):
                    m = 120 * k
                    f.append(np.prod(1 + rets[t - m:t]) - 1)
                future = np.prod(1 + rets[t + 1:t + 1 + self.horizon]) - 1
                feats.append(f)
                labels.append(future)
            feats = np.array(feats)
            labels = np.array(labels)
            all_feats.append(feats)
            all_labels.append(labels)
        X = np.vstack(all_feats)
        y = np.hstack(all_labels)
        med = np.median(y)
        y_bin = (y >= med).astype(int)
        return X, y_bin

    def train_models(self):
        X, y = self.build_features_targets()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model_rf.fit(X_train, y_train)
        self.model_lr.fit(X_train, y_train)
        prf = self.model_rf.predict(X_test)
        plr = self.model_lr.predict(X_test)
        print("RF")
        print(classification_report(y_test, prf))
        print("LR")
        print(classification_report(y_test, plr))

class RepoArbEngine:
    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def run_all(self):
        for m in self.models:
            if isinstance(m, CryptoStatArb):
                m.train_models()

def main():
    symbols = ["BTC", "ETH", "LTC", "XRP"]
    arb = CryptoStatArb(symbols)
    engine = RepoArbEngine()
    engine.add_model(arb)
    engine.run_all()

if __name__ == "__main__":
    main()
