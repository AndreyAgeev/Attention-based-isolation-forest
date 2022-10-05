# Attention-based-isolation-forest
Code for conducting experiments on article "Improved Anomaly Detection by Using the Attention-Based Isolation Forest"

# Installation 
```
$ pip install -r requirements.txt
```

# Usage
To run the code, the ***hydra*** package is used to set configurations: 

There are two modes for running the code, which are set in the **config.yaml**:
* optimization
* inference 

```
mode: "optimization"

defaults:
  - inference
  - optimization
```

The corresponding configuration files are used to set up datasets, model parameters (**isolation forest** or **attention based isolation forest** and other parametrs) and logs.