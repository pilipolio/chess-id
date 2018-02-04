import torch
from torch.autograd import Variable
from torch import nn
from torchvision import models
from torchvision import transforms


CLASSES = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR', '', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']


class ImageClassifier(nn.Module):
    LAST_LAYER_SIZE = 256 * 7 * 7

    def __init__(self, num_classes, alexnet_model):
        super(ImageClassifier, self).__init__()
        self.feature_model = alexnet_model.features
        # Freeze those weights
        for p in self.feature_model.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.LAST_LAYER_SIZE, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        features = self.feature_model(x)
        flattened_features = features.view(features.size(0), -1)
        return self.classifier(flattened_features)

    @classmethod
    def create(cls, num_classes, alexnet_pretrained=True):
        model = models.alexnet(pretrained=alexnet_pretrained)
        return cls(num_classes, model)

    @classmethod
    def load(cls, num_classes, model_path):
        model = cls.create(num_classes, alexnet_pretrained=False)
        model.load_state_dict(torch.load(model_path))
        return model


# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/9
_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

pil_image_to_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    _normalize,
    lambda i: Variable(i.unsqueeze(0))
])

