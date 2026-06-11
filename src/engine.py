import torch

from src.utils import compute_accuracy


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_num=None):
    #train the model for one full pass
    #1. forward pass
    #2. loss calculation
    #3. backpropagation
    #4. optimizer update
    #5. accuracy tracking

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #clear gradients from previous batch.
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)
        
        running_loss += loss.item() * batch_size

        correct, total = compute_accuracy(outputs, labels)
        running_correct += correct
        running_total += total

        #prints progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            current_loss = running_loss / running_total
            current_acc = running_correct / running_total

            print(
                f"Epoch {epoch_num}, Batch {batch_idx + 1}/{len(train_loader)} "
                f"| Loss: {current_loss:.4f} | Acc: {current_acc:.4f}",
                flush=True
            )

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    #eval mode
    #eval on val data

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            
            running_loss += loss.item() * batch_size

            correct, total = compute_accuracy(outputs, labels)
            running_correct += correct
            running_total += total

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc


def evaluate_with_tta(model, dataloader, criterion, device):
    #tta averages predicitons between normal and horizontally flipped img to improve predicitons with small sample

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits_original = model(images)

            #flip images horizontally
            flipped_images = torch.flip(images, dims = [3])
            
            #tensor image shape : [batch, channels, height, width]
            logits_flipped = model(flipped_images)

            #avg (original and flipped) prediction scores
            logits_average = (logits_original + logits_flipped) / 2

            loss = criterion(logits_average, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(logits_average, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc


def predict_with_tta(model, test_loader, device):
    #predict labels w/ horizontal flip imgs

    model.eval()

    all_ids = []
    all_predictions = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            logits_original = model(images)

            flipped_images = torch.flip(images, dims = [3])
            logits_flipped = model(flipped_images)

            logits_average = (logits_original + logits_flipped) / 2

            predicted = torch.argmax(logits_average, dim = 1)

            all_ids.extend(list(image_ids))
            all_predictions.extend(predicted.cpu().numpy().tolist())

    return all_ids, all_predictions