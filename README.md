# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Deliverables
	Part 1
	Image Classifier Project_gpu.html
		Completed Jupyter Notebook with run results from GPU environment

	Image Classifier Project_cpu.html
		Completed Jupyter Notebook with run results from CPU environment

   Part 2
	train.py
		Python command line app for training a new network on a dataset and saving the model as a checkpoint

	predict.py
		Python command line app for using a saved checkpoint to predict the class for an input image

Supporting files
	cat_to_name.json
	workspace_utils.py

Installation Instruction
	unzip the zip archive into a folder structured like the aipnd-project folder with datasets in the flowers folder

Run Instruction
	to run Part 1 deliverable as .ipynb, copy its cell contents into Image Classifier Project.ipynb in aip-project workspace

    run train.py and predict.py in a terminal session in the aip-project workspace.
    predict.py can run in either CPU or GPU mode, it will verify a valid checkpoint is present before starting prediction
    N.B : though it's feasible to run train.py in CPU mode, it is more practical to run it in GPU mode.

    python train.py -h
      usage: train.py [-h] [--save_dir SAVE_DIR]
                      [--arch {densenet121,densenet161,resnet18,vgg16}]
                      [-lr LEARNING_RATE] [-dout DROPOUT] [-hu HIDDEN_UNITS]
                      [-e EPOCHS] [--gpu]
                      [data_dir]

      positional arguments:
        data_dir              path to datasets

      optional arguments:
        -h, --help            show this help message and exit
        --save_dir SAVE_DIR   path to checkpoint directory
        --arch {densenet121,densenet161,resnet18,vgg16}
                              model architecture: densenet121 | densenet161 |
                              resnet18 | vgg16 (default: densenet121)
        -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                              learning rate (default: 0.001)
        -dout DROPOUT, --dropout DROPOUT
                        dropout rate (default: 0.5)                        
        -hu HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
                              hidden units, one or multiple values (comma separated)
                              enclosed in single quotes. Ex1. one value: '500' Ex2.
                              multiple values: '1000, 500'
        -e EPOCHS, --epochs EPOCHS
                              total no. of epochs to run (default: 3)
        --gpu                 train in gpu mode

      Example calls:
      Ex 1, use data_dir 'flowers': python train.py flowers
      Ex 2, use save_dir 'chksav' to save checkpoint: python train.py --save_dir chksav
      Ex 3, use densenet161 and hidden_units '1000, 500': python train.py --arch densenet161 -hu '1000, 500'
      Ex 4, set epochs to 10: python train.py -e 10
      Ex 5, set learning rate to 0.002 and dropout to 0.3: python train.py -lr 0.002 -dout 0.3
      Ex 6, train in GPU mode (subject to device capability): python train.py --gpu


    python predict.py -h
      usage: predict.py [-h] [-img IMG_PTH] [-cat CATEGORY_NAMES] [-k TOP_K] [--gpu]
                        [checkpoint]

      positional arguments:
        checkpoint            path to saved checkpoint

      optional arguments:
        -h, --help            show this help message and exit
        -img IMG_PTH, --img_pth IMG_PTH
                              path to an image file
        -cat CATEGORY_NAMES, --category_names CATEGORY_NAMES
                              path to JSON file for mapping class values to category
                              names
        -k TOP_K, --top_k TOP_K
                              no. of top k classes to print
        --gpu                 predict in gpu mode

      Example Calls
      Ex 1, use checkpoint 'chkpt.pth' in 'chksav': python predict.py chksav/chkpt.pth
      Ex 2, use top_k 4 and GPU : python predict.py --top_k 4 --gpu
      Ex 3, use img_pth 'flowers/test/91/image_08061.jpg' and cat name mapper 'cat_to_name.json' :
          python predict.py --img_pth flowers/test/91/image_08061.jpg --category_names cat_to_name.json
