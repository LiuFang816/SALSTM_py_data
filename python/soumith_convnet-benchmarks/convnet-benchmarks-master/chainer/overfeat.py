import chainer
import chainer.functions as F


class overfeat(chainer.FunctionSet):
    insize = 231

    def __init__(self):
        super(overfeat, self).__init__(
            conv1=F.Convolution2D(   3,   96, 11, stride=4),
            conv2=F.Convolution2D(  96,  256,  5, pad=0),
            conv3=F.Convolution2D( 256,  512,  3, pad=1),
            conv4=F.Convolution2D( 512, 1024,  3, pad=1),
            conv5=F.Convolution2D(1024, 1024,  3, pad=1),
            fc6=F.Linear(1024 * 6 * 6, 3072),
            fc7=F.Linear(3072, 4096),
            fc8=F.Linear(4096, 1000),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=False)
        t = chainer.Variable(y_data, volatile=False)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
