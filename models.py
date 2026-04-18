"""
models.py — Baseline ve önerilen model tanımları.

Baseline 1 : ResNet-18  (fine-tuned)
Baseline 2 : ResNet-50  (fine-tuned)
Baseline 3 : DenseNet-121 (fine-tuned)
Proposed   : ResNet-50 + MC Dropout (uncertainty-aware)
"""

import torch
import torch.nn as nn
import torchvision.models as tv

import config as cfg


# ── Yardımcı: son katmanı değiştir ───────────────────────────────────────────

def _replace_head(model: nn.Module, in_features: int, dropout_p: float = 0.0) -> nn.Module:
    """Modelin classifier head'ini değiştirir."""
    head = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, cfg.NUM_CLASSES),
    )
    return head


# ── Baseline 1: ResNet-18 ─────────────────────────────────────────────────────

def build_resnet18(pretrained: bool = cfg.PRETRAINED) -> nn.Module:
    weights = tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tv.resnet18(weights=weights)
    model.fc = _replace_head(model, model.fc.in_features, dropout_p=cfg.DROPOUT_P)
    return model


# ── Baseline 2: ResNet-50 ─────────────────────────────────────────────────────

def build_resnet50(pretrained: bool = cfg.PRETRAINED) -> nn.Module:
    weights = tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = tv.resnet50(weights=weights)
    model.fc = _replace_head(model, model.fc.in_features, dropout_p=cfg.DROPOUT_P)
    return model


# ── Baseline 3: DenseNet-121 ──────────────────────────────────────────────────

def build_densenet121(pretrained: bool = cfg.PRETRAINED) -> nn.Module:
    weights = tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tv.densenet121(weights=weights)
    in_feat = model.classifier.in_features
    model.classifier = _replace_head(model, in_feat, dropout_p=cfg.DROPOUT_P)
    return model


# ── Proposed: ResNet-50 + MC Dropout ──────────────────────────────────────────

class MCDropoutResNet50(nn.Module):
    """
    ResNet-50 tabanında MC Dropout ile belirsizlik tahmini yapan model.

    MC Dropout: test zamanında da Dropout aktif tutularak T forward pass yapılır;
    çıktıların varyansı epistemic (model) belirsizliğini gösterir.
    """

    def __init__(self, pretrained: bool = cfg.PRETRAINED, dropout_p: float = cfg.DROPOUT_P):
        super().__init__()
        weights    = tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone   = tv.resnet50(weights=weights)
        in_feat    = backbone.fc.in_features
        # Feature extractor (fc hariç)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # (B, 2048, 1, 1)
        # Classifier ile araya Dropout ekliyoruz (birden fazla katman)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, cfg.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def enable_dropout(self):
        """Tüm Dropout katmanlarını eval modunda bile aktif hale getirir."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ── Model fabrikası ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "resnet18":         build_resnet18,
    "resnet50":         build_resnet50,
    "densenet121":      build_densenet121,
    "mc_resnet50":      lambda: MCDropoutResNet50(),
}


def get_model(name: str) -> nn.Module:
    """
    İsme göre model döndürür ve cfg.DEVICE'e taşır.
    Geçerli isimler: resnet18, resnet50, densenet121, mc_resnet50
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model: '{name}'. Seçenekler: {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name]()
    return model.to(cfg.DEVICE)
