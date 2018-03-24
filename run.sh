data_dir='/home/liqing/Desktop/VizWiz_new/data/'
cd data/create_vocab
#python create_vocab.py
cd ../extract_feat
#python extract_feat.py
cd ../encode_QA
#python encode_QA.py
cd ../tf_record
#python convert.py
cd ../../
export gpu=1
export ckpt_prefix=saved_model
python train.py --ckpt_prefix=$ckpt_prefix --gpu=$gpu --max_epochs=30
python evaluate.py --ckpt_prefix=$ckpt_prefix --gpu=$gpu --out_dir=$ckpt_prefix --split=val
python evaluate.py --ckpt_prefix=$ckpt_prefix --gpu=$gpu --out_dir=$ckpt_prefix --split=test