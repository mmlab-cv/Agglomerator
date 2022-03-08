import os
import numpy as np
import warnings
import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule
from datamodules import SmallNORBDataModule, CIFAR100DataModule
from pytorch_lightning.loggers import WandbLogger

from models import Agglomerator
from utils import TwoCropTransform, count_parameters
from custom_transforms import CustomTransforms

import flags_Agglomerator
from absl import app
from absl import flags
FLAGS = flags.FLAGS


def init_all():
    warnings.filterwarnings("ignore")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # torch.backends.cudnn.deterministic = True

    pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()


def main(argv):
    init_all()
    wandb_logger = WandbLogger(project="Agglomerator", name=FLAGS.exp_name)
    wandb_logger.experiment.config.update(FLAGS)

    DataModuleWrapper = {
        "MNIST": MNISTDataModule,
        "FashionMNIST": FashionMNISTDataModule,
        "smallNORB": SmallNORBDataModule,
        "CIFAR10": CIFAR10DataModule,
        "CIFAR100": CIFAR100DataModule,
        "IMAGENET": ImagenetDataModule
    }

    if FLAGS.dataset not in DataModuleWrapper.keys():
        print("❌ Dataset not compatible")
        quit(0)

    dm = DataModuleWrapper[FLAGS.dataset](
        "./datasets", 
        batch_size=FLAGS.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        drop_last=True
    )
    
    ct = CustomTransforms(FLAGS)

    # Apply trainsforms
    if(FLAGS.supervise):
        dm.train_transforms = ct.train_transforms[FLAGS.dataset]
        dm.val_transforms = ct.test_transforms[FLAGS.dataset]
        dm.test_transforms = ct.test_transforms[FLAGS.dataset]
    else:
        dm.train_transforms = TwoCropTransform(ct.train_transforms[FLAGS.dataset])
        dm.val_transforms = TwoCropTransform(ct.test_transforms[FLAGS.dataset])
        dm.test_transforms = TwoCropTransform(ct.test_transforms[FLAGS.dataset])

    model = Agglomerator(FLAGS)

    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.load_checkpoint_dir)

    print("Total trainable parameters: ", count_parameters(model))

    if FLAGS.mode == "train":
        trainer = pl.Trainer(
            gpus=-1, 
            strategy='dp',
            max_epochs=FLAGS.max_epochs, 
            limit_train_batches=FLAGS.limit_train, 
            limit_val_batches=FLAGS.limit_val, 
            limit_test_batches=FLAGS.limit_test, 
            logger=wandb_logger,
            reload_dataloaders_every_n_epochs = 1
        )

        model = model.load_from_checkpoint(checkpoint_dir, FLAGS=FLAGS, strict=False) if FLAGS.resume_training else model

        trainer.fit(model, dm)

    elif FLAGS.mode == "test":
        model = model.load_from_checkpoint(checkpoint_dir, FLAGS=FLAGS, strict=False)
        model.configure_optimizers()
        
        dm.prepare_data()
        dm.setup()

        trainer = pl.Trainer(
            gpus=-1, 
            strategy='dp', 
            resume_from_checkpoint=checkpoint_dir, 
            max_epochs=FLAGS.max_epochs,
            limit_train_batches=FLAGS.limit_train, 
            limit_val_batches=FLAGS.limit_val, 
            limit_test_batches=FLAGS.limit_test
        )

        trainer.test(model, test_dataloaders=dm.test_dataloader())

    elif FLAGS.mode == "freeze":
        
        datasplits = [dm.train_dataloader, dm.val_dataloader, dm.test_dataloader]
        modes = ["Training", "Validation", "Test"]
        features_names = ['/features_train', '/features_val', '/features_test']
        labels_names = ['/labels_train', '/labels_val', '/labels_test']

        for i, (d, m, f, l) in enumerate(zip(datasplits, modes, features_names, labels_names)):
            model = model.load_from_checkpoint(checkpoint_dir, FLAGS=FLAGS, strict=False)
            model.configure_optimizers()

            dm.prepare_data()
            dm.setup()

            trainer = pl.Trainer(
                gpus=-1, 
                strategy='dp', 
                resume_from_checkpoint=checkpoint_dir, 
                max_epochs=FLAGS.max_epochs,
                limit_train_batches=FLAGS.limit_train, 
                limit_val_batches=FLAGS.limit_val, 
                limit_test_batches=FLAGS.limit_test
            )

            trainer.test(model, test_dataloaders=d())
            
            print(m + " features shape: ", np.array(model.features).shape)
            np.save('output/' + FLAGS.dataset + f, np.array(model.features))
            np.save('output/' + FLAGS.dataset + l, np.array(model.labels))

        model.batch_acc = 0

if __name__ == '__main__':
    app.run(main)