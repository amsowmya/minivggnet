from keras.utils import plot_model
from pyimagesearch.nn.conv import MiniVggNet

model = MiniVggNet.build(32, 32, 3, 10)
# visualization graph to disk
plot_model(model, to_file="VGGNet.png", show_shapes=True)
