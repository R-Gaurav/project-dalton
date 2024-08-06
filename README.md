# project-dalton

DALTON - Deep Local Learning via local Weights and Surrogate-Derivative Trasfer

Code at: https://github.com/R-Gaurav/project-dalton

# ******************************************************************************

## Dependency:

 	pytorch >= 1.12.1
 	hickle
 	matplotlib
 	tonic
  aedat

       OR

 	Install the requirements.txt file already in this directory, via:
 	`pip install -r requirements.txt`

# *****************************************************************************

## Command to run the code (execute the following from src/ dir):

 `OMP_NUM_THREADS=4 python execute_train_eval_net.py --dataset=CIFAR10 --epochs=50 --gpu_id=0 --scnn_arch=1 --batch_size=250 --seed=0`


 In the above command,
 	`dataset`: can be one of CIFAR10, MNIST, FMNIST for BPTT experiments,
       and DVS_GESTURE, DVS_MNIST, CIFAR10, DVS_CIFAR10 for RTRL experiments.

 	`epochs`: determine the number of training epochs

 	`gpu_id`: determines the device ID of the GPU to run the experiments.

 	`scnn_arch`: determines the architecture to run the experiments on. It
 	    can be 1, 2, 3, 4, 5 for A1, A2, A3, A4, and A5 archs.

	`batch_size`: determines the batch size of the dataset to be fed in.

 	`seed`: determines the seed value for fix the initialization of network.

# *****************************************************************************

## For the Strided-Conv mode, one has to comment the following lines:

 "stride: 1"
 "pool_size: 2"

 and uncomment:

 "stride: 2"

 from all the blocks in the files <DATASET.yaml> in `./consts/data_yaml_files/`

# *****************************************************************************

 To set the training method (SurrGD, v-TFR, DALTON) and setting (BPTT, RTRL)
 following configs can be set in `./consts/runtime_consts.py`	:

 MODEL_NAME = "SRGT_GD_SCNN" -- for SurrGD in BPTT setting.
 MODEL_NAME = "TFR_SCNN" -- for v-TFR in BPTT setting.
 MODEL_NAME = "DALTON_SCNN" -- for DALTON in BPTT setting.
 MODEL_NAME = "DALTON_RTRL_SCNN" for DALTON in RTRL setting.
 MODEL_NAME = "TFR_RTRL_SCNN" for TFR in RTRL setting.

# *****************************************************************************

 To set the value of `tau_cur`, `alpha`, and `beta` change the respective
 config in `./consts/exp_consts.py`

 TAU_CUR_LIST = [] -- for `tau_cur`.
 GAIN_LIST = [] -- for `alpha`.
 BIAS_LIST = [] -- for `beta`.

# *****************************************************************************

 Following all the above, set the directory paths in file
./consts/dir_consts.py`.

 BASE_DIR == path/where/you/have/downloaded/this/directory.

# *****************************************************************************

 NOTE: In case of OOM error, please try reducing the `batch_size` as your GPU
 might have insufficient memory.
