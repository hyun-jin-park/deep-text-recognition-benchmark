python3 train.py --sensitive --imgW 256 --imgH 32 --batch_size 32 --batch_max_length 32 --PAD \
--train_data test_lmdb/train --valid_data test_lmdb/val \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
