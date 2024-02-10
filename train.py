import torch

# Training
def train(model, data, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


# Evaluate
def evaluate(model, data):
    model.eval()
    # Get predictions
    with torch.no_grad():
      logits = model(data)
    pred = logits.argmax(dim=1)
    test_correct = pred[data.val_mask] == data.y[data.val_mask]
    test_acc = int(test_correct.sum()) / int(data.val_mask.sum())
    return test_acc

# Test
def test(model, data):
    model.eval()
    # Get predictions
    with torch.no_grad():
      logits = model(data)
    pred = logits.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc