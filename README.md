# Trading OS Value-Momentum v1.0

Nifty 50 + Nifty Next 50 Value-Momentum scanner.

## Scoring

Value = 30

Quality = 20

Momentum = 35

Base / Breakout = 15

Total = 100

Additional:

Early Momentum Score = 100

## Outputs

output/master_top30.csv

output/top10_early_momentum.csv

output/top10_value_momentum.csv

output/top10_breakout_ready.csv

output/telegram_message.txt

## Run

pip install -r requirements.txt

python run_value_momentum.py

## Important

The system ranks stocks rather than requiring
every stock to score 100.

The objective is to identify:

1. Strong value + quality
2. Strong momentum
3. Base formation
4. Pre-breakout candidates
5. Early momentum

This is a research/scanning system and does not
guarantee future returns.
