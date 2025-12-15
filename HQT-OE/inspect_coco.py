import torch
import numpy as np
from datasets.coco_instance import COCOInstance

# MODIFICA CON IL TUO PERCORSO
PATH_COCO = "/content/COCO_Local"

def analyze_structure(data, indent=0):
    space = "  " * indent
    if isinstance(data, dict):
        print(f"{space}dict keys: {list(data.keys())}")
        for k, v in data.items():
            print(f"{space}Key '{k}':")
            analyze_structure(v, indent+1)
            # Stop after first key to avoid spam
            break 
    elif isinstance(data, list):
        print(f"{space}List length: {len(data)}")
        if len(data) > 0:
            print(f"{space}First item type: {type(data[0])}")
            analyze_structure(data[0], indent+1)
    elif isinstance(data, torch.Tensor):
        print(f"{space}Tensor shape: {data.shape} | Dtype: {data.dtype}")
    else:
        print(f"{space}Value type: {type(data)}")

print("--- 🧐 INSPECTING COCO DATASET ---")
try:
    # Setup minimale
    ds_module = COCOInstance(path=PATH_COCO, img_size=(518, 518))
    ds_module.setup()
    dataset = ds_module.train_dataset
    
    # Prendi un elemento
    print(f"\nEstrazione elemento 0...")
    img, target = dataset[0]
    
    print(f"\n--- STRUTTURA TARGET (Che tipo di dati riceviamo?) ---")
    print(f"Type Root: {type(target)}")
    analyze_structure(target)

except Exception as e:
    print(f"Errore: {e}")