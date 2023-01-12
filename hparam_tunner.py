import torch.optim as optim

from utils.TrainUtils import load_dataset, create_loss_function, train_model
from models.SiameseNet import MobileNet

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import numpy as np

def load_datasets_form_file():
    return load_dataset(dataset_code='kfii')#, train_batch_size=config['batch_size'])

def tunning_step(config):
    train_dataloader, validation_dataloader = load_datasets_form_file()

    model = MobileNet(embedding_size=config['embedding_size']).to('cude')
    criterion = create_loss_function(model, config['triplet_loss_alpha'], 0.01)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    train_losses, validation_losses, train_accuracy, val_accuracy = train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, epochs=10, device='cuda')

    tune.report(loss=validation_losses[-1], accuracy=val_accuracy[-1])

def tune_mobilenet():
    config = {
        "triplet_loss_alpha": tune.choice([.1, .2, .5, .7, .9]),
        "embedding_size": tune.sample_from(lambda _: 2 ** np.random.randint(5, 8)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy"])

    result = tune.run(
        tunning_step,
        # resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

def main():
    tune_mobilenet()

if __name__ == '__main__':
    main()