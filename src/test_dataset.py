from dataset import BrainTumorDataset


dataset = BrainTumorDataset(
    "data/raw/Training"
)

print("Number of images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)

image, label = dataset[0]

print("Image type:", type(image))
print("Label:", label)