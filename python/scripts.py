def train(model, train_loader, loss_function, optimizer):
    model.train()
    device=next(model.parameters()).device
    for (images, labels) in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_accuracy(model, data_loader):
    model.eval()
    correct=0
    total=0
    device=next(model.parameters()).device
    for (images, labels) in data_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        predicted_digits=outputs.argmax(dim=1)
        correct+=int((predicted_digits==labels).sum())
        total+=labels.size(0)
    return correct/total
