import numpy as np

def create_shifted_frames(rc_data, gt_data):
    x = rc_data[:, 0 : rc_data.shape[1] - 1, :, :]
    y = gt_data[:, 1 : gt_data.shape[1], :, :]
    return x, y

def load_data(rc_path, gt_path):
    rc_dataset = np.load(rc_path)
    gt_dataset = np.load(gt_path)

    print("shape after loading:", rc_dataset.shape, gt_dataset.shape)

    # Add a channel dimension since the images are grayscale.
    rc_dataset = np.expand_dims(rc_dataset, axis=-1)
    gt_dataset = np.expand_dims(gt_dataset, axis=-1)

    # Normalize the data to the 0-1 range
    # TODO: add here better conversion
    rc_dataset = rc_dataset / 255 * 255
    gt_dataset = gt_dataset / 255 * 255


    print("shape after expand dimesnions:", rc_dataset.shape, gt_dataset.shape)
    return rc_dataset, gt_dataset


def prepare_data(rc_path, gt_path):
    # Load data
    rc_dataset, gt_dataset = load_data(rc_path, gt_path)
    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(rc_dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * rc_dataset.shape[0])]
    val_index = indexes[int(0.9 * rc_dataset.shape[0]) :]

    rc_train_dataset = rc_dataset[train_index]
    rc_val_dataset = rc_dataset[val_index]

    gt_train_dataset = gt_dataset[train_index]
    gt_val_dataset = gt_dataset[val_index]

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(rc_train_dataset, gt_train_dataset)
    x_val, y_val = create_shifted_frames(rc_val_dataset, gt_val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
    
    return x_train, y_train, x_val, y_val