from torchvision.transforms import Lambda, Compose, CenterCrop, RandAugment, AutoAugment, AutoAugmentPolicy, RandomCrop, RandomInvert, RandomPosterize, RandomSolarize, RandomResizedCrop, RandomAffine, GaussianBlur, RandomHorizontalFlip, Resize, RandomApply, ColorJitter, RandomGrayscale, RandomPerspective, RandomRotation, ToTensor, Normalize, RandomErasing, CenterCrop
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from datamodules import cifar100_normalization
from utils import TwoCropTransform

class CustomTransforms():
    def __init__(self, FLAGS):
        self.train_transforms = {}
        self.test_transforms = {}

        self.FLAGS = FLAGS

        # -------------------------------------------------------------------------------------

        self.train_transforms['MNIST'] = Compose([
            RandAugment(),
            ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1) ),
            Normalize((0.5,), (0.5,)),
        ])
        self.test_transforms['MNIST'] = Compose([
            ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1) ),
            Normalize((0.5,), (0.5,))
        ])
        
        # -------------------------------------------------------------------------------------

        self.train_transforms['FashionMNIST'] = Compose([
            RandAugment(),
            ToTensor(),
            Normalize((0.5,), (0.5,)),
        ])
        self.test_transforms['FashionMNIST'] = Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
        
        # -------------------------------------------------------------------------------------

        self.train_transforms['smallNORB'] = Compose([
            Resize((FLAGS.image_size,FLAGS.image_size)),
            RandomCrop(32, padding=0),
            RandAugment(),
            ToTensor(),
            Normalize((0.5,), (0.5,)),
        ])
        self.test_transforms['smallNORB'] = Compose([
            Resize((FLAGS.image_size,FLAGS.image_size)),
            CenterCrop(32),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
        
        # -------------------------------------------------------------------------------------

        self.train_transforms['CIFAR10'] = Compose([
            Resize((FLAGS.image_size,FLAGS.image_size)), 
            RandAugment(),
            # AutoAugment(AutoAugmentPolicy.CIFAR10),
            ToTensor(),
            cifar10_normalization(),
        ])
        self.test_transforms['CIFAR10'] = Compose([
            Resize((FLAGS.image_size,FLAGS.image_size)), 
            # RandAugment(),
            ToTensor(),
            cifar10_normalization(),
        ])
        
        # -------------------------------------------------------------------------------------

        self.train_transforms['CIFAR100'] = Compose([
            RandAugment(),
            ToTensor(),
            cifar100_normalization(),
            RandomErasing(),
        ])
        self.test_transforms['CIFAR100'] = Compose([
            Resize((FLAGS.image_size,FLAGS.image_size)), 
            ToTensor(),
            cifar100_normalization(),
        ])
        
        # -------------------------------------------------------------------------------------

        self.train_transforms['IMAGENET'] = Compose([
            Resize(FLAGS.image_size + 32),
            RandomCrop(FLAGS.image_size),
            RandAugment(),
            ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
        ])
        self.test_transforms['IMAGENET'] = Compose([
            Resize(FLAGS.image_size + 32),
            CenterCrop(FLAGS.image_size),
            ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
        ])