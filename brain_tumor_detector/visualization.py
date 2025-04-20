import matplotlib.pyplot as plt

def plot_comparison(original, prediction, title):
    """Plot original image vs model prediction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(prediction)
    ax2.set_title('Model Prediction')
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
