# Constructive Method + SGE:  A Constructive Method Based on Dynamic Solution Space for Travelling Salesman Problem
In Constructive Method, the decision changes the solution space dynamically. Our Method is an improvement for Constructive Method based on the dynamic solution space.

Our code is on the top of [AM](https://github.com/wouterkool/attention-learn-to-route) and [Sym-NCO](https://github.com/alstn12088/Sym-NCO/Sym-NCO-POMO). 

## Sym-NCO + SGE

Firstly, go to folder:
```bash
cd Sym-NCO
```

### Test


**Sym-NCO + SGE test**
```bash
python test_symsge.py
```


**Sym-NCO baseline test**
```bash
python test_symnco.py
```


**POMO baseline test**
```bash
python test_pomo.py
```


### Training

You can change "sub_graph_emb" to apply our improvement for Sym-NCO or POMO.

**Sym-NCO+SGE training**
```bash
python train_symsge.py
```

**Sym-NCO training**
```bash
python train_symnco.py
```

**POMO training**
```bash
python train_pomo.py
```

## AM + SGE

Our method can be also applied to vanilla AM model. 

Firstly, go to folder:
```bash
cd AM/
```

### Test

We use the pretrained model provided by the [AM](https://github.com/wouterkool/attention-learn-to-route).

```bash
python eval.py data/tsp/tsp100_test_nodes.pkl --model pretrained/tsp_100 --decode_strategy sample --width 1280 --eval_batch_size 1
```

### Train

The option '--SGE' applies our improvement to the AM.

**General**
```bash
python run.py --graph_size 100 --baseline rollout --run_name 'tsp100_rollout' --val_dataset data/tsp/tsp100_validation_seed1357.pkl --SGE
```

## Dependencies (Same with AM)

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)


