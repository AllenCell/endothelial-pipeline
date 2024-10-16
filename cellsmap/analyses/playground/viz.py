import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from cellsmap.analyses.playground import const, pipeline

def plot_loss_curves(trainer, train_loss, test_loss):

    path = trainer.exp.get_output_dir()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), test_loss, label='Test Loss')
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()

def get_classification_accuracy(x, xpred):
    label = x.long().cpu().numpy()
    label_pred = torch.argmax(xpred, dim=1).long().cpu().numpy()
    correct_predictions = (label == label_pred).astype(np.float32)
    accuracy = correct_predictions.sum() / len(label)
    return accuracy

def save_reconstruction_examples(trainer):

    device = trainer.exp.get_device()
    model = trainer.exp.get_architecture()
    path = trainer.exp.get_output_dir(subfolder="reconstructions")

    model.eval()
    with torch.no_grad():
        for _, test_batch in enumerate(trainer._test_loader):

            batch_curr = test_batch
            if isinstance(test_batch, list):
                batch_curr = test_batch[0]

            batch_curr = batch_curr.to(device)
            output = model(batch_curr)

            batch_next_pred = output
            if isinstance(output, tuple):
                batch_next_pred = output[0]

            acc1 = get_classification_accuracy(test_batch[2], output[1])
            acc2 = get_classification_accuracy(test_batch[3], output[2])
            print(f"Accuracy: {100*acc1:.1f}")
            print(f"Accuracy: {100*acc2:.1f}")

            for imid, (img_curr, img_next_pred) in enumerate(zip(batch_curr, batch_next_pred)):
                if imid % (const.BATCH_SIZE // 10):
                    continue

                img_curr = img_curr.cpu().numpy().squeeze()
                img_next_pred = img_next_pred.cpu().numpy().squeeze()

                if img_curr.ndim == 3:
                    img_curr = np.moveaxis(img_curr, 0, -1)
                    if img_curr.shape[-1] > 3:
                        img_curr = img_curr.mean(axis=-1)

                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                axs[0].imshow(img_curr, cmap='gray', vmin=0, vmax=1)
                axs[0].set_title('Target')
                axs[0].axis('off')
                axs[1].imshow(img_next_pred, cmap='gray', vmin=0, vmax=1)
                axs[1].set_title('Prediction')
                axs[1].axis('off')
                plt.savefig(os.path.join(path, f'crop_{imid}.png'))
                plt.close(fig)
            
            return
