# Function to display the node embedding (After performing dimensionality reduction to 2D)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(model, data):

    model.eval()
    out = model(data)

    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=data.y, cmap="Set2")
    plt.show()