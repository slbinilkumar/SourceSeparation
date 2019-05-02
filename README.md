## Source separation
This is our source separation repository. You can train our models on any MIDI corpus. We used the [Musical AI MIDI dataset](https://composing.ai/dataset). Feel free to work with a smaller dataset.

#### Parsing
You must parse your dataset once before beginning the training. Run `parse_dataset.py -h` for a list of arguments. This will create text files containing the relative path to each valid MIDI file in the dataset along with a list of instruments being played in each. One file is created for the train set and one for the test set. 

#### Training
Run `train_model.py -h` for a list of arguments. Below is an example of training configuration:

`guitar_drums_to_guitar E:\Datasets\Midi OverdrivenGuitar,Drums OverdrivenGuitar`

You will need [visdom](https://github.com/facebookresearch/visdom) for training (it is also listed in requirements.txt). 

Note that the training begins by filling the chunk pool, which might take time. The pool is filled until it reaches at least 100% of the pool size. If you expect to run the training script multiple times, you can cache the first pool to disk with the option `--quickstart`.

#### Testing
Run `test_model.py -h` for a list of arguments. 
 