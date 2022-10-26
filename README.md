This project is under review for the ICASSP-23 conference.

# Neural Audio Energy

### Submodules

This repo rely on submodules, clone it with 

```bash
git clone --recursive git@github.com:anonymous9992/neural_audio_energy.git
```

or manually fetch the submodules using

```bash
git submodule update --init
```

### Requirements
Install requirements 
```bash
pip install -r requirements.txt
```

### Training
Example of training the small melgan variant on the ljspeech dataset
```bash
python train.py melgan --variant small --dataset ljspeech --gpu 0
```
Batch size is scaled automatically to fill the GPU memory.

