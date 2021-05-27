import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

# saves a plot of train, val acc and loss
def plot(train_accuracy, val_accuracy, train_loss, val_loss, save_path,name):
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train acc')
    plt.plot(val_accuracy, color='blue', label='val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(save_path+'/'+name+'_training_accuracy.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path+'/'+name+'_training_loss.png')