import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fraud_detector/src'))

from preprocessing import load_train_data
from preprocessor import build_preprocessing_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default=ROOT / 'fraud_detector/train_data/train.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'fraud_detector/models/preprocessing.json')
    args = parser.parse_args()
    train_path = args.train.resolve()
    output_path = args.output.resolve()
    train = load_train_data(train_path)
    state = build_preprocessing_state(train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Preprocessing saved to {output_path}')


if __name__ == '__main__':
    main()
