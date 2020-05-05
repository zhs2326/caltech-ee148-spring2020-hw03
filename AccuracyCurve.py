import matplotlib.pyplot as plt

plt.plot(range(1, 31), [84,91,93,93,94,94,94,95,95,95,95,95,95,95,95,95,95,96,96,96,96,96,96,96,96,96,96,96,96,96], label='training accuracy')
plt.plot(range(1, 31), [94,96,97,97,97,97,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98], label='validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.title('training and validation accuracy without data augmentation')
plt.show()