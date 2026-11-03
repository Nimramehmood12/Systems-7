# Day 7 - Multiclass Image Classification and Transfer Learning

Transfer learning reuses visual features learned from a large dataset. Feature
extraction freezes the pretrained base and trains a new classifier; fine-tuning
unfreezes selected upper layers with a small learning rate. It is usually more
data-efficient than training from scratch. Evaluation should include classwise
metrics and visualized predictions, not accuracy alone.

Run `python main.py`. Practice solution `q18_mobilenet_transfer.py` compares a
small CNN with a frozen MobileNetV2 base. TensorFlow is required; ImageNet
weights are downloaded by Keras on first use.
