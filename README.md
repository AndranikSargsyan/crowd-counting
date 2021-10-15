# Crowd Counting

Create virtual environment
```shell
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Download [JHU Crowd++ dataset](https://drive.google.com/drive/folders/1FkdvHyAom1B2aVj6_jZpZPW01sQNiI7n) and extract into `./jhu_crowd_v2.0` path

For starting the training run
```shell
python -m crowdnet.train
```

For inference run
```shell
python -m crowdnet.predict --model-path MODEL_PATH --img-path IMAGE_PATH
```

[Download pretrained weights](https://drive.google.com/file/d/1YFxRZOiH3g5wOTj4vXCLxBSOqJknyuPk/view?usp=sharing)
