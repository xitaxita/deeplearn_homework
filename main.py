import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import glob
import random

# ==========================================
# 1. Data layer (Dataset)
# ==========================================
class PlantDataset(Dataset):
    def __init__(self, data_list, img_dir, transform=None, path_map=None):
        self.data_list = data_list
        self.transform = transform
        self.img_dir = img_dir
        # path_map is pre-scanned by Trainer and passed in to avoid repeated disk scans
        self.path_map = path_map

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_id, label = self.data_list[idx]
        img_path = self.path_map.get(str(img_id))
        try:
            image = Image.open(img_path).convert('RGB') if img_path else Image.new('RGB', (224, 224))
        except:
            image = Image.new('RGB', (224, 224))
        if self.transform: image = self.transform(image)
        return image, label

# ==========================================
# 2. Loss wrapper (addresses long-tail distribution)
# ==========================================
class LogitAdjustmentLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        cls_priors = cls_num_list / cls_num_list.sum()
        self.adjustment = tau * torch.log(cls_priors + 1e-12)

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.adjustment.to(logits.device).unsqueeze(0), target)

# ==========================================
# 3. Core training engine
# ==========================================
class PlantTrainer:
    def __init__(self, model, train_pool, val_pool, config, device):
        self.model = model
        self.train_pool = train_pool # original 300k training pool
        self.val_pool = val_pool
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=0.05)
        self.start_epoch = 0
        self.best_acc1 = 0.0
        
        # Pre-scan all image paths (time-consuming but done only once)
        self.path_map = self._scan_images()
        
        # Build Loss (uses pre-sampling distribution as prior)
        self.criterion = LogitAdjustmentLoss(config['cls_counts'])

    def _scan_images(self):
        print(f"正在深度扫描图片目录，请稍候...")
        dirs = [self.config['train_dir'], self.config['val_dir']]
        path_map = {}
        for d in dirs:
            files = glob.glob(os.path.join(d, "**", "*.j*"), recursive=True)
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                path_map[name] = f
        print(f"扫描完成，共索引 {len(path_map)} 张图片")
        return path_map

    def load_checkpoint(self):
        if os.path.exists(self.config['ckpt_path']):
            print(f">>> 发现检查点 {self.config['ckpt_path']}，正在恢复...")
            ckpt = torch.load(self.config['ckpt_path'], map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_acc1 = ckpt.get('best_acc1', 0.0)
            print(f">>> 恢复成功！从 Epoch {self.start_epoch + 1} 继续训练")
            return True
        return False

    def save_checkpoint(self, epoch, current_acc, is_best):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc1': self.best_acc1,
        }
        torch.save(state, self.config['ckpt_path'])
        if is_best:
            torch.save(self.model.state_dict(), self.config['best_weight_path'])

    def get_dynamic_loader(self):
        """Called each epoch to generate a randomly sampled DataLoader"""
        random.shuffle(self.train_pool)
        sampled_list = []
        counts = {}
        for img_id, label in self.train_pool:
            counts[label] = counts.get(label, 0)
            if counts[label] < self.config['sample_limit']:
                sampled_list.append((img_id, label))
                counts[label] += 1
        
        ds = PlantDataset(sampled_list, self.config['train_dir'], self.config['t_trans'], self.path_map)
        return DataLoader(ds, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    def train(self):
        self.load_checkpoint()
        val_ds = PlantDataset(self.val_pool, self.config['val_dir'], self.config['v_trans'], self.path_map)
        val_loader = DataLoader(val_ds, batch_size=self.config['batch_size'], num_workers=4)

        for epoch in range(self.start_epoch, self.config['epochs']):
            # 1. 动态采样训练
            self.model.train()
            train_loader = self.get_dynamic_loader()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(imgs), lbls)
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})

            # 2. 验证
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    correct += self.model(imgs).argmax(1).eq(lbls).sum().item()
                    total += lbls.size(0)
            
            acc = (correct / total) * 100
            print(f"--- Epoch {epoch+1} 验证准确率: {acc:.2f}% ---")

            # 3. 保存
            is_best = acc > self.best_acc1
            if is_best: self.best_acc1 = acc
            self.save_checkpoint(epoch, acc, is_best)
            if is_best: print(">>> 最佳模型已更新 <<<")

# ==========================================
# 4. Run main
# ==========================================
def main():
    # Paths and configuration
    CONF = {
        'data_dir': 'data',
        'train_dir': 'data/images_train',
        'val_dir': 'data/images_val',
        'ckpt_path': 'weights/last_checkpoint.pth',
        'best_weight_path': 'weights/best_plantnet_model.pt',
        'sample_limit': 70,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 20
    }
    os.makedirs('weights', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. Preprocess metadata
    with open(os.path.join(CONF['data_dir'], 'plantnet300K_metadata.json'), 'r') as f:
        meta = json.load(f)
    
    sids = sorted(list(set([str(v['species_id']) for v in meta.values()])))
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    
    train_pool, val_pool = [], []
    cls_counts = [0] * len(sids)
    for k, v in meta.items():
        sid = str(v['species_id'])
        if v['split'] == 'train':
            train_pool.append((k, sid_to_idx[sid]))
            cls_counts[sid_to_idx[sid]] += 1
        elif v['split'] == 'val':
            val_pool.append((k, sid_to_idx[sid]))

    # B. Transform definitions
    CONF['t_trans'] = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    CONF['v_trans'] = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    CONF['cls_counts'] = cls_counts

    # C. Start trainer
    model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=len(sids)).to(device)
    
    trainer = PlantTrainer(model, train_pool, val_pool, CONF, device)
    trainer.train()

    # D. Save mapping
    with open("weights/id_mapping.json", "w") as f:
        json.dump(sid_to_idx, f)

if __name__ == "__main__":
    main()