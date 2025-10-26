import matplotlib.pyplot as plt
#from TensorFlow.keras.callbacks import History

def plot_history(history, path_figures:str='/figures/', plot_counter:int=None, start:int=0):
    """Visualization of the model trainin process

    Args:
        history (_tf.keras.callbacks.History_): learnig history of the model
        path_figures (str, optional): path to the image folder. Defaults to '/figures/'.
        plot_counter (int, optional): image number. Defaults to None.
    """
    
    #f1_sc = history.history['f1']
    mse = history.history['mse'][start:]
    loss = history.history['loss'][start:]

    #f1_sc_val = history.history['val_f1'] 
    val_mse = history.history['val_mse'][start:]
    val_loss = history.history['val_loss'][start:]

    epochs = range(len(loss))#f1_sc

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax[0].plot(epochs, loss, 'b', label='Training')
    ax[0].plot(epochs, val_loss, 'r', label='Validation')
    ax[0].set_xlabel('Epoch', size=11)
    ax[0].set_ylabel('Loss', size=11)
    ax[0].set_title('Loss', size=12)
    ax[0].legend()

    ax[1].plot(epochs, mse, 'b', label='Training')
    ax[1].plot(epochs, val_mse, 'r', label='Validation')
    ax[1].set_xlabel('Epoch', size=11)
    ax[1].set_ylabel('MSE value', size=11)
    ax[1].set_title(f"MSE error", size=12)
    ax[1].legend()
    plt.tight_layout();
    if plot_counter:
        fig.suptitle(f"Fig.{plot_counter} - Model evaluation", y=0.05, fontsize=14)
        
        fig.savefig(path_figures + f'/fig_{plot_counter}.png')
    else:
        plot_counter = 1
        fig.suptitle(f"Fig.{plot_counter} - Model evaluation", y=-0.05, fontsize=14)
        # fig.show(); #- do not call for correct logging