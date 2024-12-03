**EFTGAN**
we develop a generative network framework: Elemental Features enhanced and Transferring corrected data augmentation
in Generative Adversarial Networks (EFTGAN). Combining the elemental convolution technique with Generative Adversarial Networks (GAN),
EFTGAN provides a robust and efficient approach for generating data containing elemental and structural information that can be used not only
for data augmentation to improve model accuracy, but also for prediction when the structures are unknown. 
Applying this framework to the FeNiCoCrMn/Pd high-entropy alloys, we successfully improve the prediction accuracy
in a small data set and predict the concentrationdependent formation energies, lattices,
and magnetic moments in quinary systems. 

**Contents of documents：**
The folder HAE_Data contains all the data for the model.
The folder datasets contains the data loading datasets for the training of each model.
The folder models contains the network structure of each model.
MLP_transfer.py is for training the transfer augmentation model and predicted result.
get_ib.py is used to extract interaction blocks from the trained ECNe.
get_describer.py is used to downscale the interaction block.
model.py defines the generic class for the project's model.
model_test.py is the test file for model development.
trainer_agunet.py is used to train the augmentation model.
trainer_gan.py is used to train training InfoGAN models.
trainer_heanet.py is used to train ECNet with single task.
trainer_heanet_mtl_HEA.py is used to train ECNet with mult-task

**Environment Requirements:**
The basic environment is the PyTorch, and CUDA for your GPU machines. The projects use the graph to model pairwise relations between nodes, 
and the graph representation is then combined with the message passing networks. Thus, the framework is also based on the torch_geometric, 
where its installation could be refered to the official website [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html]
(https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 
For other packages, you can easily install them via :
pip install -r package.txt 

**Workflow:**

The workflow of our model starts with a complete ECNet model: The files for training ECNet use the console command.
Please enter the command:

python trainer_heanet_mtl_HEA.py --task etot emix eform --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --tower_h1 128 --tower_h2 64 --batch_size 128 --epochs 500 --split_type 0 -t

to train ECNet model.
Meaning of the parameters in the command：

- mtl: multi-target learning
- 1, 2, 3, ... , 6: indicates the number of tasks
- HEA: high entropy alloy
- 300, 500: the size of epochs during training.
- best: the model with the best results on the validation set
- etot/ef... :task-specific
- 128: Dimension of the elemental feature vector.
- 4+5/2+3: Whether the task is 2+3 or 4+5 epochs.
- a0/a1/a2/a3: type of partitioning of the dataset, refer to the load_hea_data code. a Use advance preprocessed files, which data are used as training, tests have been
  preprocessed in advance to the corresponding file species. a0: All data are divided into training and testing datasets according to the ratio in advance. a1: 234 sets of meta
  data for training, 5 sets of metadata for testing. a2: 23 training, 45 testing. a3: 45 sets of metadata training, 23 sets of metadata testing.
- b0/b1/b2/b3: Segmentation type of the dataset. Refer to load_hea_data_single_file code. b Distinguish between split training, test validation in the code.
  b0: Conducted in 45 multi-homogenised alloys, dividing training and testing according to the ratio. b1: Conducted in 23 low-homogenised alloys. b2: Conducted in 2+3+4+5-homogenised alloys.

Reference values for the accuracy of models trained using the above commands:
the score of task 0 is 0.07146676629781723

the score of task 1 is 0.028152920305728912

the score of task 2 is 0.03343312442302704

Afterwards the trained model is called and the interaction blocks are obtained from it. 
Please enter the command:
python getib.py --task etot emix eform --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --tower_h1 128 --tower_h2 64 --batch_size 1 --epochs 500 --processed_data --split_type 0 -p

Use get_describer.py to to downscale the interaction block. This file can be run directly

Use trainer_gan.py to train InfoGAN model. This file can be run directly.
And use the trained InfoGAN to generate the data.

Use trainer_agunet.py to train the directly augmented model.
Please enter the command:
python trainer_agunet.py --task etot emix eform --hidden_channels 128 --n_filters 64 --n_interactions 3 --is_validate --tower_h1 128 --tower_h2 64 --batch_size 128 --epochs 800 --split_type 0 -t

Use MLP_transfer.py to train the transfer augmentation model. This file can be run directly.
Modify the TRAIN or TEST parameter in the main function for model training and prediction.
