python3 train.py --sensitive --imgW 256 --imgH 32 --batch_size 48 --batch_max_length 32 --PAD \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth --lr 0.5 \
--num_iter 3000000 --valInterval 30000  \
--train_data lmdb_card/train --valid_data lmdb_card/val \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
