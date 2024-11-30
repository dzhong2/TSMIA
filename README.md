This is an anonymous repo for paper: "Membership Inference on Well-generalized Time Series Classification Models"

### How to use 

#### step 1: Train target model

`train_resnet.py`, `train_fcn.py` and `train_informer.py` includes the code for model training.

#### step 2: Train shadow model

`python3 shadow_model_training.py --dataset NetFLow --model resnet` will train shadow model on shadow dataset

#### step 3: reference model training

`python3 train_out_models.py' --dataset NetFLow --model resnet` will train reference models that removes each sample

#### step 4: derive attack feature and launch attack

`python3 Attack.py --dataset NetFLow --model resnet --num-ref 10 --attack_options XIXS` will run launch attack with an option within (XIXS, XI, XS)
