#python3 train.py --sensitive --imgW 256 --imgH 32 --batch_size 32 --batch_max_length 32 --PAD --lr 2 \
#--train_data lmdb_card/train --valid_data lmdb_card/val \
#--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
#--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth \

python3 train.py --sensitive --imgW 256 --imgH 32 --batch_size 128 --batch_max_length 32 --lr 0.1 \
--train_data card_data/train --valid_data card_data/val \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth \
--data_filtering_off --workers 16 \
--num_iter 3000000 --valInterval 10000 \
--select_data trdg-unity --batch_ratio 0.9-0.1 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
