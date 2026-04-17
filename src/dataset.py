import os
import random
import numpy as np
# import cv2
from PIL import Image
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm


def load_mha(filepath):
    return sitk.GetArrayFromImage(sitk.ReadImage(filepath))


def normalize_mr(mr_array, mask_array):
    mr   = mr_array.astype(np.float32)
    mask = mask_array.astype(bool)
    mr[~mask] = 0.0
    foreground = mr[mask]
    p1         = np.percentile(foreground, 1)
    p99        = np.percentile(foreground, 99)
    mr         = np.clip(mr, p1, p99)
    mean       = foreground.mean()
    std        = foreground.std() + 1e-8
    mr         = (mr - mean) / std
    mr[~mask]  = 0.0
    return mr


def normalize_ct(ct_array, mask_array, ct_min=-1000, ct_max=3000):
    ct        = ct_array.astype(np.float32)
    mask      = mask_array.astype(bool)
    ct        = np.clip(ct, ct_min, ct_max)
    ct        = (ct - ct_min) / (ct_max - ct_min)
    ct        = ct * 2.0 - 1.0
    ct[~mask] = -1.0
    return ct


# class SynthRADDatasetOnTheFly(Dataset):
#     def __init__(self, patient_ids, anatomy, data_root,
#                  split="train", augment=False,
#                  image_size=256, min_mask_sum=100):

#         self.data_root    = data_root
#         self.anatomy      = anatomy
#         self.split        = split
#         self.augment      = augment
#         self.image_size   = image_size
#         self.min_mask_sum = min_mask_sum
#         self.samples      = []

#         for pid in tqdm(patient_ids, desc=f"Indexing {anatomy} {split}"):
#             if ":" in pid:
#                 anat, patient_id = pid.split(":")
#             else:
#                 anat       = anatomy
#                 patient_id = pid

#             patient_path = os.path.join(data_root, anat, patient_id)
#             mr_path      = os.path.join(patient_path, "mr.mha")
#             mask_path    = os.path.join(patient_path, "mask.mha")

#             if not os.path.exists(mr_path):
#                 print(f"WARNING: Missing {mr_path}")
#                 continue

#             mask_array = load_mha(mask_path)
#             for z in range(mask_array.shape[0]):
#                 if mask_array[z].sum() >= self.min_mask_sum:
#                     self.samples.append((anat, patient_id, z))

#         print(f"  {len(self.samples)} valid slices indexed.")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         anat, patient_id, slice_idx = self.samples[idx]

#         patient_path = os.path.join(self.data_root, anat, patient_id)
#         mr_array     = load_mha(os.path.join(patient_path, "mr.mha"))
#         ct_array     = load_mha(os.path.join(patient_path, "ct.mha"))
#         mask_array   = load_mha(os.path.join(patient_path, "mask.mha"))

#         mr_norm = normalize_mr(mr_array, mask_array)
#         ct_norm = normalize_ct(ct_array, mask_array)

#         n        = mr_norm.shape[0]
#         prev_idx = max(0, slice_idx - 1)
#         next_idx = min(n - 1, slice_idx + 1)

#         mr_stack = np.stack([
#             mr_norm[prev_idx],
#             mr_norm[slice_idx],
#             mr_norm[next_idx],
#         ], axis=0).astype(np.float32)

#         ct_stack = np.stack([
#             ct_norm[prev_idx],
#             ct_norm[slice_idx],
#             ct_norm[next_idx],
#         ], axis=0).astype(np.float32)

#         mask_slice = mask_array[slice_idx].astype(np.float32)

#         # mr_resized = np.stack([
#         #     cv2.resize(mr_stack[c], (self.image_size, self.image_size),
#         #                interpolation=cv2.INTER_LINEAR)
#         #     for c in range(3)
#         # ], axis=0)

#         # ct_resized = np.stack([
#         #     cv2.resize(ct_stack[c], (self.image_size, self.image_size),
#         #                interpolation=cv2.INTER_LINEAR)
#         #     for c in range(3)
#         # ], axis=0)

#         # mask_resized = cv2.resize(
#         #     mask_slice,
#         #     (self.image_size, self.image_size),
#         #     interpolation=cv2.INTER_NEAREST
#         # )


#         mr_resized = np.stack([
#             np.array(Image.fromarray(mr_stack[c]).resize(
#                 (self.image_size, self.image_size), Image.BILINEAR))
#             for c in range(3)
#         ], axis=0)

#         ct_resized = np.stack([
#             np.array(Image.fromarray(ct_stack[c]).resize(
#                 (self.image_size, self.image_size), Image.BILINEAR))
#             for c in range(3)
#         ], axis=0)

#         mask_resized = np.array(Image.fromarray(mask_slice).resize(
#             (self.image_size, self.image_size), Image.NEAREST))


#         mr   = torch.tensor(mr_resized,   dtype=torch.float32)
#         ct   = torch.tensor(ct_resized,   dtype=torch.float32)
#         mask = torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0)

#         if self.augment:
#             mr, ct, mask = self.apply_augmentation(mr, ct, mask)

#         return {
#             "mr":      mr,
#             "ct":      ct,
#             "mask":    mask,
#             "patient": patient_id,
#             "slice":   slice_idx,
#         }

#     def apply_augmentation(self, mr, ct, mask):
#         if random.random() > 0.5:
#             mr = TF.hflip(mr); ct = TF.hflip(ct); mask = TF.hflip(mask)
#         if random.random() > 0.5:
#             mr = TF.vflip(mr); ct = TF.vflip(ct); mask = TF.vflip(mask)
#         angle = random.uniform(-10, 10)
#         mr    = TF.rotate(mr,   angle)
#         ct    = TF.rotate(ct,   angle)
#         mask  = TF.rotate(mask, angle)
#         if random.random() > 0.5:
#             shift = random.uniform(-0.1, 0.1)
#             mr    = torch.clamp(mr + shift, mr.min(), mr.max())
#         return mr, ct, mask





class SynthRADDatasetOnTheFly(Dataset):
    def __init__(self, patient_ids, anatomy, data_root,
                 split="train", augment=False,
                 image_size=256, min_mask_sum=100):

        self.data_root    = data_root
        self.anatomy      = anatomy
        self.split        = split
        self.augment      = augment
        self.image_size   = image_size
        self.min_mask_sum = min_mask_sum
        self.samples      = []
        self.volume_cache = {}          # ← NEW: cache full 3D volumes

        for pid in tqdm(patient_ids, desc=f"Indexing {anatomy} {split}"):
            if ":" in pid:
                anat, patient_id = pid.split(":")
            else:
                anat       = anatomy
                patient_id = pid

            patient_path = os.path.join(data_root, anat, patient_id)
            mr_path      = os.path.join(patient_path, "mr.mha")
            mask_path    = os.path.join(patient_path, "mask.mha")

            if not os.path.exists(mr_path):
                print(f"WARNING: Missing {mr_path}")
                continue

            try:
                mask_array = self._safe_load_mha(mask_path)
            except Exception as e:
                print(f"ERROR loading mask for {patient_id}: {e}")
                continue

            for z in range(mask_array.shape[0]):
                if mask_array[z].sum() >= self.min_mask_sum:
                    self.samples.append((anat, patient_id, z))

        print(f"  {len(self.samples)} valid slices indexed.")

    def _safe_load_mha(self, filepath):
        """Safe load with clear error message"""
        if filepath in self.volume_cache:
            return self.volume_cache[filepath]

        try:
            print(f"Loading volume: {filepath}")   # ← you will see this
            img = sitk.ReadImage(filepath)
            arr = sitk.GetArrayFromImage(img)
            self.volume_cache[filepath] = arr
            return arr
        except Exception as e:
            print(f"CRITICAL ERROR reading {filepath}")
            print(f"Exception: {type(e).__name__}: {e}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anat, patient_id, slice_idx = self.samples[idx]

        patient_path = os.path.join(self.data_root, anat, patient_id)

        try:
            mr_array   = self._safe_load_mha(os.path.join(patient_path, "mr.mha"))
            ct_array   = self._safe_load_mha(os.path.join(patient_path, "ct.mha"))
            mask_array = self._safe_load_mha(os.path.join(patient_path, "mask.mha"))
        except Exception as e:
            print(f"❌ SKIPPING bad sample → Patient: {patient_id} | Slice: {slice_idx}")
            print(f"   Error: {e}")
            # Return a dummy zero tensor so training doesn't crash
            dummy = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return {
                "mr": dummy, "ct": dummy,
                "mask": torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32),
                "patient": patient_id,
                "slice": slice_idx,
            }

        mr_norm = normalize_mr(mr_array, mask_array)
        ct_norm = normalize_ct(ct_array, mask_array)

        n        = mr_norm.shape[0]
        prev_idx = max(0, slice_idx - 1)
        next_idx = min(n - 1, slice_idx + 1)

        mr_stack = np.stack([mr_norm[prev_idx], mr_norm[slice_idx], mr_norm[next_idx]], axis=0)
        ct_stack = np.stack([ct_norm[prev_idx], ct_norm[slice_idx], ct_norm[next_idx]], axis=0)
        mask_slice = mask_array[slice_idx].astype(np.float32)

        # Resize with PIL (your original code)
        mr_resized = np.stack([
            np.array(Image.fromarray(mr_stack[c]).resize((self.image_size, self.image_size), Image.BILINEAR))
            for c in range(3)
        ], axis=0)

        ct_resized = np.stack([
            np.array(Image.fromarray(ct_stack[c]).resize((self.image_size, self.image_size), Image.BILINEAR))
            for c in range(3)
        ], axis=0)

        mask_resized = np.array(Image.fromarray(mask_slice).resize(
            (self.image_size, self.image_size), Image.NEAREST))

        mr   = torch.tensor(mr_resized,   dtype=torch.float32)
        ct   = torch.tensor(ct_resized,   dtype=torch.float32)
        mask = torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0)

        if self.augment:
            mr, ct, mask = self.apply_augmentation(mr, ct, mask)

        return {
            "mr":      mr,
            "ct":      ct,
            "mask":    mask,
            "patient": patient_id,
            "slice":   slice_idx,
        }

    def apply_augmentation(self, mr, ct, mask):
        if random.random() > 0.5:
            mr = TF.hflip(mr); ct = TF.hflip(ct); mask = TF.hflip(mask)
        if random.random() > 0.5:
            mr = TF.vflip(mr); ct = TF.vflip(ct); mask = TF.vflip(mask)
        angle = random.uniform(-10, 10)
        mr    = TF.rotate(mr, angle)
        ct    = TF.rotate(ct, angle)
        mask  = TF.rotate(mask, angle)
        if random.random() > 0.5:
            shift = random.uniform(-0.1, 0.1)
            mr    = torch.clamp(mr + shift, mr.min(), mr.max())
        return mr, ct, mask








def make_dataloader(patient_ids, anatomy, data_root,
                    split, batch_size, num_workers, image_size=256):
    dataset = SynthRADDatasetOnTheFly(
        patient_ids = patient_ids,
        anatomy     = anatomy,
        data_root   = data_root,
        split       = split,
        augment     = (split == "train"),
        image_size  = image_size,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        pin_memory  = True,
    )