# ift6289-project

This project is for the class IFT6289. It implements the code from the paper [Leveraging Visual Question Answering to Improve Text-to-Image
Synthesis](https://doi.org/10.48550/arXiv.2010.14953) by Stanislav Frolov, Shailza Jolly, Jörn Hees and Andreas Dengel.

# Dependencies
You can install the dependencies in the file [environment.yml](environment.yml) which was created for anaconda.

Once the dependencies are installed, you can also install the module pytorch fid by doing:
````shell
pip install pytorch-fid
````

# Downloading data
You can download the preprocessed data for COCO by following the instructions in the [README of AttnGAN](src/attnGAN/README.md). Follow their instructions for unzipping the files as well.

You can then download the images as well as the VQA 2.0 questions and answers from [this link](https://visualqa.org/download.html). You only need the training and validation annotations, questions and image files from 2017.
Rename the files ``v2_mscoco_train2014_annotations.json`` and `v2_mscoco_val2014_annotations.json` into `mscoco_train2014_annotations.json` and `mscoco_val2014_annotations.json` respectively.

You can also download the pretrained DAMSM model from the [README file of AttnGAN](src/attnGAN/README.md) in the pretrained model section. Download the one for COCO. You can save it in `src/attnGAN/DAMSMencoders/`

Please follow this structure for the data:
````
.                           # in your data folder
├── images_train            # Only the images from the training set
├── images_val              # Only the images from the validation set
├── test                    # contains filenames.pickle from the validation set (you can find this in the preprocessed data of COCO)
├── train                   # contain filenames.pickle from the training set
├── text                    # contains the text files from the preprocessed data of COCO (both validation and training set)
├── captions.pickle         # contains the vocabulary of the captions (found in the preprocessed COCO data)
├── mscoco_train2014_annotations.json
├── mscoco_val2014_annotations.json
├── OpenEnded_mscoco_train2014_questions.json
└── OpenEnded_mscoco_val2014_questions.json
````

# Configuration
There are two configuration files that need modification: [coco_attn2.yml](src/attnGAN/code/cfg/coco_attn2.yml) and [config.py](src/attnGAN/code/vqa/config.py).

### coco_attn2.yml
For this file, you need to change the following inputs:
````yaml
DATA_DIR: '' # path to your data folder
OUTPUT_DIR: '' # path to where you want to save your models and generated images
TRAIN:
  NET_E: '../DAMSMencoders/coco/text_encoder100.pth' # path to where you saved the pretrained DAMSM model
````
The rest depends on your experiment.

**Note on SNAPSHOT_INTERVAL** This number represent the frequency at which you save your model. For example, if the number is equal to 2, your model will be saved every 2 epochs. 

### config.py

For this file, you only need to change `qa_path` for the filepath to where your data are (should be the same as the one in `DATA_DIR` of `coco_attn2.yml`)

# Training

### Baseline: AttnGAN

You can train a baseline model that only trains on COCO dataset and not VQA by using this command in the terminal (we consider that you are in `src/attnGAN/code`):
````shell
python main.py --cfg cfg/coco_attn2.yml --manualSeed 10 --gpu 0
````
This basically trains the original AttnGAN model.

### Leveraging VQA
If you want to train using VQA 2.0 dataset, you can then use:
````shell
python main.py --cfg cfg/coco_attn2.yml --manualSeed 10 --gpu 0 --with_vqa
````

If you want to record your experiment on comet_ml, you can add an environment variable names `COMET_API_KEY` which contains your API key from Comet ml. Then, you can simply add the argument `--comet` to your run.

### Start training from a previous epoch
In the case that you want to restart training your model, you can modify the `NET_G` parameter in [coco_attn2.yml](src/attnGAN/code/cfg/coco_attn2.yml). For example, if I have a model trained up until epoch 50 called and I want to continue its training, I can modify NET_G to the filepath where this model is saved. Your model should always be saved in the `OUTPUT_DIR` that you have given in [coco_attn2.yml](src/attnGAN/code/cfg/coco_attn2.yml). `NET_G` takes the generator filepath. Your model should then be saved under the name of `netG_epoch_50.pth.tar`.
````yaml
TRAIN:
  NET_G: 'netG_epoch_50.pth.tar' # filepath to where your model is saved, put the generator file
````
Once this is done, you can start training like you would a new model, but it will start at this epoch. The optimizers will also restart where they were left off.

# Evaluate model

Every epoch, your model should be evaluated on the validation set and generate the inception score. It should also save about a hundred of generated images so that you can analyse them.

Once training is done, you can evaluated your model on the validation set for the Inception Score, FID and VQA accuracy by running the following command:
````shell
python main.py --cfg cfg/coco_attn2.yml --manualSeed 10 --gpu 0
````
which is the same as the training command.

Before running the command, you need to modify [coco_attn2.yml](src/attnGAN/code/cfg/coco_attn2.yml).
Change the following parameters:
````yaml
B_VALIDATION: True # this one has to be added to the file

TRAIN:
  FLAG: False
````

This should print the Inception Score, FID score and VQA accuracy.