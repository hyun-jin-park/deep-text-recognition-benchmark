python3 train.py --sensitive --imgW 256 --imgH 32 --batch_size 32 --batch_max_length 32 --PAD \
--train_data lmdb_card/train --valid_data lmdb_card/val \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
