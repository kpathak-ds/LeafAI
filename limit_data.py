import os
import random

def limit_images(folder_path, limit=100):
    if not os.path.exists(folder_path):
        print(f"Path not found: {folder_path}")
        return

    for label in os.listdir(folder_path):
        path = os.path.join(folder_path, label)

        if not os.path.isdir(path):
            continue

        images = os.listdir(path)

        if len(images) > limit:
            to_delete = random.sample(images, len(images) - limit)

            for img in to_delete:
                os.remove(os.path.join(path, img))

    print(f"Done limiting: {folder_path}")

# Correct paths
limit_images("PlantVillage", 100)
limit_images("olid", 100)