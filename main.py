import torch 
from utils.dataloader import planetoid_dataset
from utils.visualizer import visualize
from train import train, test
from model.model import GNN
from torch.utils.tensorboard import SummaryWriter

def main():

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Device: {device}")

    # Loading the dataset
    print("__________________\n\n Downloading Data\n__________________")
    dataset = planetoid_dataset()

    # Init the model
    model = GNN(n_features=dataset.num_features, n_classes=dataset.num_classes)

    # Visualize un-trained model embedding
    visualize(model, dataset)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    # Create a SummaryWriter for logging
    writer = SummaryWriter()

    print("__________________\n\n  Training \n__________________")

    # Training
    n_epochs = 25
    for epoch in range(n_epochs):
      loss = train(model, dataset, optimizer, loss_function)
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


    # Evaluate:
    test_acc = test(model, dataset)
    print(f'Test Accuracy: {test_acc:.4f}')



if __name__ == "__main__":
    main()