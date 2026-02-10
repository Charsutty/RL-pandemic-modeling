# RL-pandemic-modeling

This repo shows a study of RL methods to respond to a problem of pandemic propagation and regulation through action of $u_{conf}$ for lockdown and $u_{vacc}$ for vaccination level. We use the model S(Sane),E(Exposed),I(Infected),R(Restablished),D(Deceased).

Our implementation relies on the [Gymnasium Representation](https://gymnasium.farama.org/).

You can setup your environment with 
```bash
pip install -r requirements.txt
```

The notebook `implementation.ipynb` will run you through our experiments. All the code has been externalized in python module files to make the experience cleaner.

