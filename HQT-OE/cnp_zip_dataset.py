import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, get_worker_info

class CNPZipDataset(Dataset):
    def __init__(self, zip_path: str):
        self.zip_path = Path(zip_path)
        self._zips = {}

        with zipfile.ZipFile(self.zip_path) as z:
            self.images = sorted([n for n in z.namelist() if n.startswith("images/") and n.endswith(".png")])
            self.masks  = sorted([n for n in z.namelist() if n.startswith("masks/")  and n.endswith(".png")])

        assert len(self.images) == len(self.masks) and len(self.images) > 0, \
            f"Mismatch images={len(self.images)} masks={len(self.masks)}"

    def _get_zip(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        if wid not in self._zips:
            self._zips[wid] = zipfile.ZipFile(self.zip_path)
        return self._zips[wid]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        z = self._get_zip()

        # image -> float [3,H,W] in [0,1]
        with z.open(self.images[idx]) as f:
            img = Image.open(BytesIO(f.read())).convert("RGB")
            img.load()
        img_t = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            .view(img.size[1], img.size[0], 3).numpy()
        ).permute(2,0,1).float() / 255.0

        # mask -> long [H,W] in {0,1}
        with z.open(self.masks[idx]) as f:
            m = Image.open(BytesIO(f.read())).convert("L")
            m.load()
        m_t = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(m.tobytes()))
            .view(m.size[1], m.size[0]).numpy()
        ).long()
        m_t = (m_t > 127).long()

        return img_t, m_t, 1