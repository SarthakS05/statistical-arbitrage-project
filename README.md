# Statistical Arbitrage Experiments

An educational Python project centered on a cryptocurrency statistical-arbitrage experiment, with additional BERT, augmented-reality, budget-forcing, and BitTorrent examples.

## Main workflow

Running `statistical_arbitrage.py` requests minute-level price history from CryptoCompare for BTC, ETH, LTC, and XRP, builds return-based features, trains random-forest and logistic-regression classifiers, and prints classification reports. It requires internet access and depends on CryptoCompare's endpoint availability and rate limits. No API key is configured in this project.

The BERT examples download pretrained models from Hugging Face when instantiated. The augmented-reality example accesses a camera. The BitTorrent example is incomplete: its torrent parser is a placeholder and does not implement a full client.

This is research and demonstration code, not a production trading system or financial advice. Running the main script makes external network requests and trains models.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the main experiment from the repository root:

```bash
python statistical_arbitrage.py
```