Batch size should equal to number of block sizes -> try 10 layers with batch size 10

## Setup (Ubuntu 22.04) 

### 1. Clone repository

### 2. Set up git lfs (not necessary atm)
1. [Git lfs](https://git-lfs.com/)

    ```
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    ```

    ```
    sudo apt-get install git-lfs
    ```

    ```
    git lfs install
    ```

### 3. Install dependencies
1. Install from Pipfile:

    ```
    pipenv install
    ```

2. Activate environment:
    ```
    pipenv shell
    ```

### 4. Set up config.yml
#### 1. Copy config.yml.example to config.yml
#### 2. Fill in the required fields

## Train

```
python -m lstm.train
```

# TODO:
1. Try bidirectional training
2. Change input shape of convLSTM network
4. Set-up multilayer ConvLSTM