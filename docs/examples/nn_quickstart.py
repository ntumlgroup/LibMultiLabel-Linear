from datetime import datetime

from libmultilabel.nn import data_utils
from libmultilabel.nn.nn_utils import init_device, init_model, init_trainer, set_seed


run_name = 'rcv1-KimCNN-example_{}'.format(
    datetime.now().strftime('%Y%m%d%H%M%S'))
checkpoint_dir = f'runs/{run_name}'
set_seed(1337)


# Step 1. Load dataset and build dictionaries.
datasets = data_utils.load_datasets(data_dir='data/rcv1', val_size=0.2)
classes = data_utils.load_or_build_label(datasets)
word_dict = data_utils.load_or_build_text_dict(
    dataset=datasets['train'],
    embed_file='glove.6B.300d')

# Step 2. Initialize model with network config.
network_config = {
    "dropout": 0.2,
    "filter_sizes": [2, 4, 8],
    "num_filter_per_size": 128
}
model = init_model(model_name='KimCNN',
                   network_config=network_config,
                   classes=classes,
                   word_dict=word_dict,
                   monitor_metrics=['P@1'])

# Step 3. Initialize trainier.
trainer = init_trainer(checkpoint_dir=checkpoint_dir,
                       val_metric='P@1')

# Step 4. Create data loaders.
device = init_device()
train_loader = data_utils.get_dataset_loader(
    data=datasets['train'],
    word_dict=word_dict,
    classes=classes,
    device=device,
    batch_size=32
)
val_loader = data_utils.get_dataset_loader(
    data=datasets['val'],
    word_dict=word_dict,
    classes=classes,
    device=device,
    batch_size=32
)

#  Step 5. Train a model from scratch.
trainer.fit(model, train_loader, val_loader)
