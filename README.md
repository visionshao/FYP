# FYP
My FYP proj

# To run the resnet_finetune
- Put fer2013.csv in "data" folder
- python fer2013.py
- python preprocess_fer2013.py
- nohup python -u resnet_finetune.py  2>&1 | tee -a result.txt &

# To test the model
- nohup python -u test.py  2>&1 | tee -a test_result.txt &
